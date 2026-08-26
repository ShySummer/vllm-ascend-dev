# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

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
