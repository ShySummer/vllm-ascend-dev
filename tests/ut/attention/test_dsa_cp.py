# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel import dsa_cp
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPImpl,
    AscendDSAPCPImpl,
    AscendDSAPCPMetadataBuilder,
)
from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


def test_qli_metadata_uses_cpu_lengths_without_device_scalar_reads():
    from vllm_ascend.attention.context_parallel.dsa_cp import (
        AscendDSACPMetadataBuilder,
    )

    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.compressor_ratio = 4
    builder.common_ratio_to_sas_metadata = {}
    builder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            index_n_heads=64,
            index_head_dim=128,
            index_topk=2048,
        )
    )
    builder.seqused_q = torch.zeros(2, dtype=torch.int32)
    builder.qli_seqused_k = torch.zeros(2, dtype=torch.int32)
    builder.qli_cmp_residual_k = torch.zeros(2, dtype=torch.int32)
    builder.req_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    query_start = torch.tensor([0, 2, 5], dtype=torch.int32)
    seq_lens = torch.tensor([12, 31], dtype=torch.int32)
    expected = torch.arange(1024, dtype=torch.int32)
    with (
        patch.object(
            torch.ops._C_ascend,
            "npu_quant_lightning_indexer_v2_metadata",
            return_value=expected,
            create=True,
        ) as metadata_op,
        patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("device scalar read"),
        ),
    ):
        result = builder._build_qli_metadata(
            query_start,
            seq_lens,
            2,
            max_seqlen_q=3,
            max_seqlen_k=31,
        )
        builder._build_qli_metadata(
            query_start,
            seq_lens,
            2,
            max_seqlen_q=3,
            max_seqlen_k=31,
        )
    assert metadata_op.call_count == 1
    assert metadata_op.call_args.kwargs["max_seqlen_q"] == 3
    assert metadata_op.call_args.kwargs["max_seqlen_k"] == 31 // 4
    assert builder.qli_seqused_k.tolist() == [3, 7]
    assert builder.qli_cmp_residual_k.tolist() == [0, 3]
    assert metadata_op.call_args.kwargs["layout_k"] == "PA_BBND"
    assert torch.equal(result, expected)


class TestAscendDSACPLayerMetadata:
    def test_routes_by_cache_prefix(self):
        impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
        impl.compress_ratio = 4
        impl.swa_cache_layer = SimpleNamespace(prefix="swa_cache")
        impl.compressor = SimpleNamespace(state_cache=SimpleNamespace(prefix="compressor.state_cache"))
        impl.indexer = SimpleNamespace(
            k_cache=SimpleNamespace(prefix="indexer.k_cache"),
            compressor=SimpleNamespace(state_cache=SimpleNamespace(prefix="indexer.compressor.state_cache")),
        )
        attention_metadata = object()
        compressor_state_metadata = object()
        indexer_cache_metadata = object()
        indexer_state_metadata = object()
        swa_metadata = object()

        metadata: Any = {
            "layer": attention_metadata,
            "compressor.state_cache": compressor_state_metadata,
            "indexer.k_cache": indexer_cache_metadata,
            "indexer.compressor.state_cache": indexer_state_metadata,
            "swa_cache": swa_metadata,
        }
        layer_metadata = impl._get_layer_metadata("layer", metadata)

        assert layer_metadata.swa is swa_metadata
        assert layer_metadata.compressor_cache is attention_metadata
        assert layer_metadata.compressor_state is compressor_state_metadata
        assert layer_metadata.indexer_cache is indexer_cache_metadata
        assert layer_metadata.indexer_state is indexer_state_metadata


class TestAscendDSAPCPGraphPadding:
    @pytest.mark.parametrize(
        "full_graph_mode,expected_num_tokens",
        [
            (False, 6),
            (True, 8),
        ],
    )
    def test_restore_indices_match_graph_mode(self, full_graph_mode, expected_num_tokens):
        """FULL graphs replay the fixed padded batch, so restore indices cover
        every padded row; other modes restore only the actual token extent,
        keeping placeholder rows out of the cache updates entirely."""
        builder = AscendDSAPCPMetadataBuilder.__new__(AscendDSAPCPMetadataBuilder)
        builder._hidden_restore_idx_buffer = torch.full((16,), -1, dtype=torch.int64)

        global_batch = SimpleNamespace(num_tokens=6, num_tokens_after_padding=8, is_dummy=False)
        hidden_restore_idx = torch.arange(10, 16, dtype=torch.int64)
        pcp_context = AscendPCPAttentionContext(
            global_batch=global_batch,
            global_block_tables=(),
            global_slot_mappings=torch.zeros(0, dtype=torch.int64),
            hidden_restore_idx=hidden_restore_idx,
        )

        prepared = builder._prepare_graph_pcp_context(pcp_context, full_graph_mode=full_graph_mode)

        assert prepared.hidden_restore_idx.shape[0] == expected_num_tokens
        assert torch.equal(prepared.hidden_restore_idx[:6], hidden_restore_idx)
        if full_graph_mode:
            assert torch.equal(prepared.hidden_restore_idx[6:], torch.zeros(2, dtype=torch.int64))

    def test_swa_cache_update_consumes_metadata_as_built(self):
        """cos/sin/slot_mapping are built at the restored batch length (padded
        for FULL graphs, actual tokens otherwise), so the cache update passes
        them to the rotary/scatter kernels without any trimming."""
        impl = AscendDSAPCPImpl.__new__(AscendDSAPCPImpl)
        impl.nope_head_dim = 4
        impl.rope_head_dim = 2
        impl.head_dim = 6
        impl.vllm_config = SimpleNamespace()
        impl.wkv = MagicMock(side_effect=lambda x: x)
        impl.kv_norm = MagicMock(side_effect=lambda x: x)

        num_tokens = 3
        cos = torch.randn(num_tokens, 2)
        sin = torch.randn(num_tokens, 2)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
        req_metadata = SimpleNamespace(
            cos={"layer": cos},
            sin={"layer": sin},
            slot_mapping=slot_mapping,
        )
        rotary = MagicMock()
        scatter = MagicMock()
        with (
            patch(
                "vllm_ascend.attention.dsa_v1._require_req_metadata",
                return_value=req_metadata,
            ),
            # The op registers lazily at worker startup, so create the
            # attribute on the op namespace for this CPU-only UT.
            patch("torch.ops._C_ascend.inplace_partial_rotary_mul", rotary, create=True),
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.get_dsa_attn_kv_plan",
                return_value=SimpleNamespace(dsa_kv_compress_scatter=scatter),
            ),
        ):
            impl._update_global_swa_cache(
                "layer",
                torch.randn(num_tokens, 6),
                torch.randn(1),
                MagicMock(),
            )

        rotary.assert_called_once()
        assert rotary.call_args.args[1] is cos
        assert rotary.call_args.args[2] is sin
        scatter.assert_called_once()
        assert scatter.call_args.args[2] is slot_mapping


@pytest.mark.parametrize("fail_at", [None, "project", "copy"])
def test_o_projection_keeps_weight_lifetime_in_caller(monkeypatch, fail_at):
    events = []
    metadata = SimpleNamespace(attn_state=dsa_cp.AscendAttentionState.PrefillNoCache)
    tensor = torch.ones(2, 4)

    def project(value, *, full_gather_wo_a_enabled):
        assert full_gather_wo_a_enabled
        events.append("project")
        if fail_at == "project":
            raise RuntimeError("projection failed")
        return value

    class Output:
        def __setitem__(self, index, value):
            events.append("copy")
            if fail_at == "copy":
                raise RuntimeError("copy failed")

    impl = SimpleNamespace(
        tp_size=2,
        enable_dsa_cp_full_o_proj=True,
        _get_layer_metadata=lambda *args: SimpleNamespace(compressor_cache=metadata),
        _forward=lambda *args: tensor,
        _restore_tp_head_layout=lambda *args, **kwargs: tensor,
        _switch_o_proj_to_full_weight=lambda: events.append("full"),
        _switch_o_proj_to_local_weight=lambda: events.append("local"),
        _forward_o_proj=project,
    )
    monkeypatch.setattr(dsa_cp, "wait_for_kv_layer_from_connector", lambda *args: None)
    monkeypatch.setattr(dsa_cp, "maybe_save_kv_layer_to_connector", lambda *args: events.append("save"))
    output = Output()
    if fail_at:
        with pytest.raises(RuntimeError, match="failed"):
            AscendDSACPImpl.forward(impl, "layer", tensor, (), {}, output)
        assert events[-1] == "local"
        assert "save" not in events
    else:
        assert AscendDSACPImpl.forward(impl, "layer", tensor, (), {}, output) is output
        assert events == ["full", "project", "copy", "local", "save"]


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("fp8_attention", [False, True])
@pytest.mark.parametrize("with_output", [False, True])
def test_o_projection_common_computation_uses_active_weight_layout(monkeypatch, full, fp8_attention, with_output):
    def batch_matmul(value, weight, **kwargs):
        return torch.bmm(value.transpose(0, 1), weight).transpose(0, 1)

    monkeypatch.setattr(dsa_cp.torch_npu, "npu_transpose_batchmatmul", batch_matmul, raising=False)
    monkeypatch.setattr(dsa_cp, "_has_weight_scale", lambda layer: True)
    method = object.__new__(dsa_cp.AscendUnquantizedLinearMethod)
    groups = 2 if full else 1
    value = torch.arange(4 * groups * 3, dtype=torch.float32).reshape(4, groups * 3)
    weight = torch.arange(groups * 3 * 2, dtype=torch.float32).reshape(groups, 3, 2)
    wo_b = torch.arange(groups * 2 * 5, dtype=torch.float32).reshape(groups * 2, 5)
    impl = SimpleNamespace(
        n_group=2,
        n_local_groups=1,
        support_fp8_attention=fp8_attention,
        wo_a=SimpleNamespace(quant_method=method),
        _get_batched_wo_a_weight=lambda count: weight,
        _apply_wo_b=lambda value, gathered: value @ wo_b,
    )
    expected = torch.einsum("tgh,ghr->tgr", value.reshape(4, groups, 3), weight).reshape(4, -1) @ wo_b
    output = torch.empty_like(expected) if with_output else None
    actual = AscendDSACPImpl._forward_o_proj(impl, value, output, full_gather_wo_a_enabled=full)
    if with_output:
        assert actual is output
    torch.testing.assert_close(actual, expected)

    def test_update_graph_params_is_noop_like_dsa(self):
        # FULL_DECODE_ONLY replay calls this on the impl class.
        AscendDSACPImpl.update_graph_params(None, None, 1)


def test_zero_graph_padding_rows_clears_stale_seq_lens_and_block_table():
    from vllm_ascend.attention.context_parallel.dsa_cp import (
        AscendDSACPMetadataBuilder,
    )

    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.seq_lens = torch.tensor([17, 4096, 128], dtype=torch.int32)
    builder.seq_lens_cpu = torch.tensor([17, 4096, 128], dtype=torch.int32)
    builder.block_table = torch.tensor([[3, 4], [9, 8], [7, 6]], dtype=torch.int32)
    builder.start_pos_prefill = torch.tensor([16, 99, 50], dtype=torch.int32)

    actual = builder._zero_graph_padding_rows(num_reqs=3, num_actual_reqs=1)

    assert actual == 1
    assert torch.equal(builder.seq_lens, torch.tensor([17, 0, 0], dtype=torch.int32))
    assert torch.equal(builder.seq_lens_cpu, torch.tensor([17, 0, 0], dtype=torch.int32))
    assert torch.count_nonzero(builder.block_table[1:]).item() == 0
    assert torch.equal(builder.start_pos_prefill, torch.tensor([16, 0, 0], dtype=torch.int32))


def test_zero_graph_padding_rows_noop_when_no_pad():
    from vllm_ascend.attention.context_parallel.dsa_cp import (
        AscendDSACPMetadataBuilder,
    )

    builder = AscendDSACPMetadataBuilder.__new__(AscendDSACPMetadataBuilder)
    builder.seq_lens = torch.tensor([4, 5], dtype=torch.int32)
    builder.seq_lens_cpu = torch.tensor([4, 5], dtype=torch.int32)
    builder.block_table = torch.tensor([[1], [2]], dtype=torch.int32)
    builder.start_pos_prefill = torch.tensor([3, 4], dtype=torch.int32)

    actual = builder._zero_graph_padding_rows(num_reqs=2, num_actual_reqs=2)

    assert actual == 2
    assert torch.equal(builder.seq_lens, torch.tensor([4, 5], dtype=torch.int32))
    assert torch.equal(builder.block_table, torch.tensor([[1], [2]], dtype=torch.int32))
