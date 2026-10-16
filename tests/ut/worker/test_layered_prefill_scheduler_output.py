# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_ascend.worker import model_runner_v1 as runner_module
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize("include_one_time_updates", [False, True])
@pytest.mark.parametrize("updates", [None, {}, {"a": ([10],), "b": ([20],), "other": ([30],)}])
def test_layered_subbatch_uses_current_scheduler_output_fields(include_one_time_updates, updates):
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [SimpleNamespace(req_id="a"), SimpleNamespace(req_id="b")]
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["a", "b"],
        resumed_req_ids={"a", "b"},
        new_token_ids=[[11], [22]],
        all_token_ids={"a": [1, 11], "b": [2, 22]},
        new_block_ids=[([10],), ([20],)],
        num_computed_tokens=[3, 7],
        num_output_tokens=[1, 2],
    )
    output.num_scheduled_tokens = {"a": 3, "b": 1}
    output.total_num_scheduled_tokens = 4
    output.scheduled_spec_decode_tokens = {"a": [11], "b": [22]}
    output.scheduled_encoder_inputs = {"a": [0], "b": [1]}
    output.num_invalid_spec_tokens = {"a": 1, "b": 2}
    output.block_table_updates = updates
    one_time = {
        "finished_req_ids": ({"finished"}, set()),
        "preempted_req_ids": ({"preempted"}, set()),
        "free_encoder_mm_hashes": (["hash"], []),
        "new_block_ids_to_zero": ([99], None),
        "kv_cache_block_copies": ([object()], None),
        "scheduled_encoder_input_stats": (object(), None),
        "kv_connector_metadata": (object(), None),
        "ec_connector_metadata": (object(), None),
        "ec_manager_metadata": (object(), None),
    }
    for name, (value, _) in one_time.items():
        setattr(output, name, value)
    plan = object()
    result = NPUModelRunner._subset_scheduler_output(
        output,
        ["b"],
        layered_plan=plan,
        include_one_time_updates=include_one_time_updates,
    )
    assert isinstance(result, SchedulerOutput)
    assert not hasattr(result, "partial_tail_offloads")
    assert result.block_table_updates == (None if updates is None else {k: v for k, v in updates.items() if k == "b"})
    assert result.kv_connector_block_state is None
    assert result.layered_prefill_plan is plan
    assert result.num_scheduled_tokens == {"b": 1}
    assert result.total_num_scheduled_tokens == 1
    assert [r.req_id for r in result.scheduled_new_reqs] == ["b"]
    cached = result.scheduled_cached_reqs
    assert cached.req_ids == ["b"]
    assert cached.resumed_req_ids == {"b"}
    assert cached.new_token_ids == [[22]]
    assert cached.all_token_ids == {"b": [2, 22]}
    assert cached.new_block_ids == [([20],)]
    assert cached.num_computed_tokens == [7]
    assert cached.num_output_tokens == [2]
    assert result.scheduled_spec_decode_tokens == {"b": [22]}
    assert result.scheduled_encoder_inputs == {"b": [1]}
    assert result.num_invalid_spec_tokens == {"b": 2}
    for name, (value, empty) in one_time.items():
        if include_one_time_updates:
            assert getattr(result, name) is value
        else:
            assert getattr(result, name) == empty
    assert output.block_table_updates is updates
    assert output.num_scheduled_tokens == {"a": 3, "b": 1}
    assert output.scheduled_cached_reqs.req_ids == ["a", "b"]
    if updates:
        result.block_table_updates.clear()
        assert set(output.block_table_updates) == {"a", "b", "other"}


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("lmhead_tp", [False, True])
def test_layered_sample_uses_current_sampler_interface(monkeypatch, trace, lmhead_tp):
    monkeypatch.setenv("VLLM_LAYERED_PREFILL_TRACE", "1" if trace else "0")
    monkeypatch.setattr(runner_module, "lmhead_tp_enable", lambda: lmhead_tp)
    log = MagicMock()
    monkeypatch.setattr(runner_module, "logger", log)
    metadata = SimpleNamespace(top_k=torch.tensor([2, 2]))
    sampled = SimpleNamespace(sampled_token_ids=torch.tensor([[1], [2]]))
    # An explicit callable spec has no obsolete prepare_sampling method.
    sampler = MagicMock(spec=lambda logits, sampling_metadata: None, return_value=sampled)
    batch = SimpleNamespace(
        update_async_output_token_ids=MagicMock(),
        sampling_metadata=metadata,
        num_reqs=2,
        req_ids=["a", "b"],
    )
    runner = SimpleNamespace(
        input_batch=batch,
        sampler=sampler,
        _executing_layered_subbatch=True,
    )
    logits = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    result = NPUModelRunner._sample(runner, logits, None)
    assert result is sampled
    batch.update_async_output_token_ids.assert_called_once_with()
    sampler.assert_called_once()
    args = sampler.call_args.kwargs
    assert args["sampling_metadata"] is metadata
    torch.testing.assert_close(args["logits"], logits[:2] if lmhead_tp else logits)
    log.info.assert_not_called()


@pytest.mark.parametrize("fail_prefill", [False, True])
def test_layered_pp_subbatches_select_tokens_and_restore_state(monkeypatch, fail_prefill):
    synchronize = MagicMock()
    device_sync = MagicMock(side_effect=AssertionError("device-wide sync forbidden"))
    stream = SimpleNamespace(synchronize=synchronize)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda: stream, synchronize=device_sync),
        raising=False,
    )
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"d": 1, "p": 3}
    output.total_num_scheduled_tokens = 4
    output.layered_prefill_plan = SimpleNamespace(prefill_req_ids=["p"])
    output.finished_req_ids = {"finished"}
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.requests = {"d": object(), "p": object()}
    main_batch = SimpleNamespace(remove_request=MagicMock())
    prefill_batch = SimpleNamespace(remove_request=MagicMock())
    runner.input_batch = main_batch
    runner.layered_prefill_model_adapter = object()
    runner.speculative_config = None
    runner._get_layered_prefill_input_batch = lambda _n: prefill_batch
    runner._capture_layered_sampling_masks = lambda: ([], 0, [])
    calls = []

    def execute(sub_output, intermediate):
        is_prefill = sub_output.layered_prefill_plan is not None
        calls.append(
            (
                runner._layered_moe_comm_token_count,
                sub_output.num_scheduled_tokens,
                sub_output.finished_req_ids,
            )
        )
        assert runner._executing_layered_subbatch
        assert intermediate is None
        if is_prefill and fail_prefill:
            raise RuntimeError("mock prefill failure")
        return IntermediateTensors(
            {
                "hidden_states": torch.full(
                    (sub_output.total_num_scheduled_tokens, 2),
                    2.0 if is_prefill else 1.0,
                )
            }
        )

    runner.execute_model = execute
    if fail_prefill:
        with pytest.raises(RuntimeError, match="mock prefill failure"):
            runner._execute_layered_step(output, None)
    else:
        packed = runner._execute_layered_step(output, None)
        d_out, p_out = runner._split_layered_pp_intermediate(packed)
        torch.testing.assert_close(d_out["hidden_states"], torch.ones(1, 2))
        torch.testing.assert_close(p_out["hidden_states"], torch.full((3, 2), 2.0))
    assert calls == [(None, {"d": 1}, {"finished"}), (4, {"p": 3}, set())]
    assert runner.input_batch is main_batch
    assert runner._layered_moe_comm_token_count is None
    assert not runner._executing_layered_subbatch
    prefill_batch.remove_request.assert_called_once_with("finished")
    assert synchronize.call_count == (1 if fail_prefill else 2)
    device_sync.assert_not_called()
