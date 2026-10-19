# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from vllm.compilation import breakable_cudagraph
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import async_tensor_h2d as async_copy_to_gpu
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.eplb_utils import step_eplb_after
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.kv_connector import KVConnector
from vllm.v1.worker.gpu.model_runner import (
    BatchReqState,
    ExecuteModelState,
    GPUModelRunner,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import (
    MoECommType,
    get_mc2_tokens_capacity,
    override_mrv2_in_profile_run,
    select_moe_comm_method,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)
from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackendImpl,
    fill_mxfp_v_scale_cache,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.flashmla import prepare_flashmla_batch_req_state
from vllm_ascend.attention.mla_v1 import AscendMLABackend, all_mla_use_flashmla
from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.core.profiling_chunk_predictor import (
    _finish_profiling_chunk_timing,
    _start_profiling_chunk_timing,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    bind_dspark_context_receiver,
    configure_dspark_kv_transfer,
    find_dspark_context_connector,
    find_dspark_prefix_connector,
    get_pd_dspark_aux_layer_ids,
    send_dspark_prefill_kv,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_kv import (
    apply_dspark_resident_kv_specs,
    get_resident_dspark_layer_names,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    allocate_kv_offload_topk_profile_buffers,
    init_sparse_kv_offload_manager,
)
from vllm_ascend.models.deepseek_v41.cache_config import uses_a5_packed_cache
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.utils import (
    is_c8_mxfp_kv_quant,
    is_deepseek_v41,
    lmhead_tp_enable,
    lmhead_tp_max_num_logits,
    lmhead_tp_pad_rows,
    set_potential_max_tokens,
    should_skip_allreduce_across_dp_group,
)
from vllm_ascend.worker.device_metadata import TargetDeviceMetadata
from vllm_ascend.worker.utils import disable_compilation
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_state,
    skip_ring_state_update,
)
from vllm_ascend.worker.v2.eplb import AscendEPLBController
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.kv_connector import MambaCopyKVConnector
from vllm_ascend.worker.v2.kvpp import KVPPRuntime
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager
from vllm_ascend.worker.v2.pp_transport import (
    bypass_upstream_spec_pp_guard,
    resolve_spec_pp_support,
    restore_pp_after_upstream_init,
    use_legacy_spec_pp,
)
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import (
    AscendV2KVBlockZeroer,
    C8KVBlockZeroer,
    prepare_v41_dummy_ring_state,
    prepare_v41_source_rope,
    torch_cuda_wrapper,
)


from types import SimpleNamespace
from vllm.distributed.parallel_state import get_pp_group
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import round_up
from vllm.v1.core.layered_prefill import (
    LayeredForwardOutput,
    LayeredPrefillStateStore,
    make_pp_aligned_layer_group_ranges,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.model_runner import sort_batch_req_ids
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.utils import enable_dsa_cp, enable_sp
from vllm_ascend import envs as envs_ascend
from vllm_ascend.worker.layered_prefill_single_forward import LayeredPrefillGroupExecutor
from vllm_ascend.worker.v2.layered_prefill import (
    LayeredPPHandlerCapture,
    LayeredV2ExecuteModelState,
    concat_req_frontiers,
    detach_execute_model_state,
    empty_layered_prefill_counters,
    pad_activation_rows,
    split_d_p_req_ids,
    store_req_frontiers,
    subset_scheduler_output,
)
from vllm_ascend.worker.v2.layered_prefill_async import LayeredPrefillAsyncOutput
from vllm_ascend.worker.v2.layered_prefill_nonowner import (
    can_compact_internal_prefill_payload,
    run_internal_prefill_fence,
    run_nonowner_prefill,
)
from vllm_ascend.worker.v2.layered_prefill_pp_stage import LayeredV2PPStageForward
from vllm_ascend.worker.v2.layered_prefill_single_forward import LayeredV2Buffers, LayeredV2SingleForward


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    # vLLM #51718 overlays hybrid Attention/Mamba groups in one standardized
    # backing allocation. Ascend MRV2 preserves that allocation and exposes
    # backend-specific views; MLA may use a page-strided fused/component view.
    supports_standardized_shared_kv_backing = True

    execute_model_state: ExecuteModelState | None
    kv_connector: KVConnector
    max_num_reqs: int

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Keep exact host counts until all active consumers are known.
        self._skip_spec_cpu_length_sync = False
        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        self.sparse_kv_offload_manager = None
        self.kvpp = KVPPRuntime()
        # Adaptive verification uses this flag to apply FIA-specific query
        # boundary and sequence length padding during FULL graph execution.
        self.use_fia = False
        self._uses_flashmla = False
        # FusedMoE can be constructed by the parent initializer and reads this
        # capacity while setting up MC2 communication.
        set_potential_max_tokens(vllm_config)
        parallel_config = vllm_config.parallel_config

        # Only release versions need PP hidden during upstream initialization.
        spec_pp_support = resolve_spec_pp_support(vllm_config)
        with torch_cuda_wrapper():
            with bypass_upstream_spec_pp_guard(vllm_config, spec_pp_support) as pp_disabled:
                super().__init__(vllm_config, device)
            if pp_disabled:
                restore_pp_after_upstream_init(self, vllm_config)
        # Native PP owns token broadcast/writeback; only releases use our packing.
        # Legacy Spec+PP transport (0.28/0.29 only); deleted when 0.30+ is the floor.
        self.use_spec_pp = spec_pp_support is not None and use_legacy_spec_pp()
        # These FIA models need post-rejection host counts on every PP stage.
        # TODO: Remove this extra PP sync when FIA and its metadata builders
        # use device lengths instead of exact CPU lengths.
        self.sync_spec_pp_cpu_counts = (
            self.use_pp
            and self.num_speculative_steps > 0
            and self.model_config.architecture
            in (
                "KimiLinearForCausalLM",
                "KimiK3ForCausalLM",
                "KimiK3ForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
            )
        )
        # These draft heads consume target aux states collected across PP ranks.
        if spec_pp_support is not None and spec_pp_support.needs_aux_hidden_states:
            self.use_aux_hidden_state_outputs = True

        # Only a real split (size > 1) exchanges across ranks, and graph dispatch keeps it aligned.
        ftpc = self.ascend_config.finegrained_tp_config
        self._finegrained_tp_requires_graph = (
            ftpc.oproj_tensor_parallel_size > 1 or ftpc.mlp_tensor_parallel_size > 1
        ) and self.dp_size > 1

        self.use_aclgraph = (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and (
                self.compilation_config.mode == CompilationMode.VLLM_COMPILE
                or breakable_cudagraph.is_breakable_cudagraph_enabled()
            )
            and not self.model_config.enforce_eager
        )
        self.eplb = AscendEPLBController(
            parallel_config,
            device,
            self.ascend_config.eplb_config if parallel_config.enable_eplb else None,
        )

        self.update_stream = None
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream = torch.npu.Stream()

        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.req_states
        del self.input_buffers
        del self.speculator

        # we define AscendEagleSpeculator in vllm_ascend.worker.v2.spec_decode.eagle.speculator
        # init_speculator will return AscendEagleSpeculator when eagle is used.
        # so here we just call init_speculator to reinitialize speculator.
        self.speculator: AscendEagleSpeculator | None = None
        self.pd_dspark_aux_layer_ids: tuple[int, ...] = ()
        self._dspark_prefill_progress: dict[str, tuple[str, int]] = {}
        if self.speculative_config is not None and self.is_last_pp_rank:
            self.speculator = init_speculator(self.vllm_config, self.device)
            # Shared update_stream: main model (ModelAclGraphManager) and draft
            # (Eagle/DFlash/DSpark AclGraphManager) all use this same stream.
            self.speculator.update_stream = self.update_stream

        # AscendRequestState has extra `num_computed_tokens_cpu` attribute.
        # so reinitialize req_states here.
        self.req_states: AscendRequestState = AscendRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        if self.use_spec_pp:
            from vllm_ascend.patch.worker.patch_v2.patch_spec_pp import (
                install_upstream_spec_pp_protocol,
            )

            assert self.pp_handler is not None
            install_upstream_spec_pp_protocol(self.pp_handler, self.req_states, self.num_speculative_steps)
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )

        # Pinned D2H staging for corrected device state after spec rejection.
        # The authoritative host state is the shared NumPy/torch RequestState view.
        self.num_computed_tokens_event = torch.npu.Event()
        self.num_computed_tokens_stream = torch.npu.Stream()
        self.num_computed_tokens_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        # NOTE: In GPUModelRunner, decode_query_len is initialized in load_model(),
        # +1 is hardcoded here but not in vllm.
        self.decode_query_len = self.num_speculative_steps + 1
        # Set _mc2_tokens_capacity and _reserved_mc2_mask for MoE communication optimization.
        # TODO: remove set_cos_and_sin (together with update_cos_sin) when mla can properly handle cos/sin internally
        set_cos_and_sin(vllm_config, self.max_num_reqs, self.decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        set_potential_max_tokens(vllm_config)

        # msprobe dump (same as v1 NPUModelRunner; not via DFX).
        dump_cfg = self.ascend_config.dump_config_path
        self.debugger = None
        if dump_cfg is not None:
            self._debugger_started = False
            if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
                from msprobe.pytorch import PrecisionDebugger

                self.debugger = PrecisionDebugger(dump_cfg)
            else:
                try:
                    from msprobe.pytorch import AclGraphDumper
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to import AclGraphDumper from msprobe. "
                        "Please install/rebuild msprobe with aclgraph_dump enabled."
                    ) from exc

                self.debugger = AclGraphDumper(dump_cfg)

        # Layered Prefill V2: second buffer set for the P sub-batch.  V2's
        # prepare_inputs writes through self.input_buffers views, so D and P
        # must not share storage.  See layered_prefill_v2_migration_plan.md §3.1.
        layered_cfg = self.ascend_config.scheduler_config.layered_prefill_config
        self._layered_prefill_enabled = bool(layered_cfg.enabled)
        self._boundary_prefill_enabled = self._layered_prefill_enabled and (
            getattr(layered_cfg, "boundary_policy", "off") == "pp_transition"
        )
        if self._layered_prefill_enabled and (
            getattr(layered_cfg, "fuse_mixed_batch", False) or getattr(layered_cfg, "same_layer_batch", False)
        ):
            raise NotImplementedError(
                "Model Runner V2 layered prefill does not support fuse_mixed_batch or same_layer_batch"
            )
        self._single_forward = bool(getattr(layered_cfg, "single_forward", True))
        self._layered_group_executor = LayeredPrefillGroupExecutor(self)
        self._layered_single_forward_executor = LayeredV2SingleForward(self, self._layered_group_executor)
        self._layered_pp_stage_executor = None
        self._layered_nonowner_fastpath = True
        self._layered_compute_num_groups = 0
        self._layered_decode_buffers = LayeredV2Buffers()
        layered_ext = getattr(self.ascend_config.scheduler_config, "layered_prefill_extensions", None)
        self._skip_relay_stages = bool(getattr(layered_ext, "skip_relay_stages", False))
        # True after load_model wires the D→P adapter (PP=1 and PP>1).
        self._layered_prefill_v2_ready = False
        self.layered_prefill_state = LayeredPrefillStateStore()
        self.layered_prefill_model_adapter = None
        # Bytes of the cross-group frontier that profile's high-water mark
        # did not already include. Subtracted from the KV budget in
        # NPUWorker.determine_available_memory.
        self.layered_frontier_profile_bytes = 0
        self._layered_skip_pp_decode_update = False
        # Set when a step is pure relay, so sample_tokens can answer without
        # an ExecuteModelState.
        self._layered_relay_output: ModelRunnerOutput | None = None
        self._layered_input_buffers: AscendInputBuffers | None = None
        self.layered_prefill_counters = empty_layered_prefill_counters()
        if self._layered_prefill_enabled:
            # Phase 1 schedules a single P request; size by max query tokens.
            # Keep max_num_reqs aligned with the main runner so query_start_loc
            # padding helpers stay valid if the scheduler grows past k=1.
            self._layered_input_buffers = AscendInputBuffers(
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self.max_num_tokens,
                device=self.device,
            )

    @property
    def pcp_manager_cls(self) -> type[AscendPCPManager]:
        return AscendPCPManager

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        aux_layers = get_pd_dspark_aux_layer_ids(self.vllm_config)
        super().load_model(load_dummy_weights, *args, **kwargs)
        self.pd_dspark_aux_layer_ids = aux_layers
        # In cudagraph modes the dumper must be started before graph capture so
        # that aclgraph capture/replay is instrumented (v1 parity: v1 starts
        # the dumper at the end of load_model).
        if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            self._start_dump_data()

        if self._layered_prefill_enabled:
            from vllm_ascend.models.layered_prefill import (
                create_layered_prefill_model_adapter,
            )

            try:
                self.layered_prefill_model_adapter = create_layered_prefill_model_adapter(self.model)
            except TypeError as error:
                raise RuntimeError("The loaded model does not support layered prefill") from error
            # Orchestration (V2 milestone) is wired; flip the fail-closed gate.
            self._layered_prefill_v2_ready = True

    def _restore_replicated_draft_target_states(self) -> None:
        """Restore target states consumed by a replicated PCP draft."""
        state = self.execute_model_state
        pcp_manager = self.pcp_manager
        if (
            state is None
            or pcp_manager is None
            or not self.is_last_pp_rank
            or not getattr(self.speculator, "replicated_pcp", False)
        ):
            return

        get_hidden_states = getattr(
            self.model,
            "get_mtp_target_hidden_states",
            None,
        )
        if get_hidden_states is not None:
            mtp_target_hidden_states = get_hidden_states()
            if mtp_target_hidden_states is not None:
                pcp_manager.restore_hidden_state_buffer(mtp_target_hidden_states)

        # vLLM main captures draft_hidden_states before maybe_restore_pcp_for_sampling,
        # so a replicated draft would read the PCP-local target output. Restore
        # it to the global layout up front. aux_hidden_states need no handling
        # here: upstream sample_tokens (#56107) already restores them, per
        # tensor, before speculator.propose.
        if state.hidden_states is not None:
            state = state._replace(hidden_states=pcp_manager.restore_hidden_states(state.hidden_states))
            # Tell restore_for_sampling to skip its second all-gather for this
            # step. The layout is tracked explicitly because a length match is
            # ambiguous under piecewise/FULL graphs: every rank pads its local
            # batch to the same global padded length.
            pcp_manager._sampling_hidden_restored = True
        self.execute_model_state = state

    def _sample_tokens_normal(self, grammar_output):
        if (
            lmhead_tp_enable()
            and self.prompt_logprobs_worker is not None
            and self.prompt_logprobs_worker.uses_prompt_logprobs.any()
        ):
            # The prompt-logprobs worker issues a second compute_logits with
            # unpadded rows that desyncs the LM-head collectives and hangs.
            raise NotImplementedError("prompt_logprobs is not supported with lmhead TP.")

        pcp_manager = self.pcp_manager
        if pcp_manager is not None and not self.is_last_pp_rank and self.execute_model_state is not None:
            assert isinstance(pcp_manager, AscendPCPManager)
            # The last PP stage restores PCP outputs to the global request
            # layout before sampling. Non-last stages do not own final hidden
            # states, but their PP receive/postprocess path must use that same
            # global request layout rather than PCP-local segment rows.
            self.execute_model_state = self.execute_model_state._replace(
                input_batch=pcp_manager.global_batch,
            )

        # The pre-restore marker is scoped to this sampling step; stale state
        # from a previous step must not suppress the upstream all-gather.
        if pcp_manager is not None and isinstance(pcp_manager, AscendPCPManager):
            pcp_manager._sampling_hidden_restored = False
        self._restore_replicated_draft_target_states()
        output = super().sample_tokens(grammar_output)
        if self.use_spec_pp and self.is_last_pp_rank:
            assert self.pp_handler is not None
            self.pp_handler.broadcast_drafts()
        # Close the per-step msprobe dump cycle on the normal last-PP-rank
        # path (v1 parity: v1 finalizes at the end of sample_tokens).
        self._finalize_dump_data()
        return output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        relay_output = self._layered_relay_output
        if relay_output is not None:
            # This stage only passed the activation on.  Nothing sampled, but
            # every scheduled request still needs a row in the output.
            self._layered_relay_output = None
            return relay_output
        state = self.execute_model_state
        if isinstance(state, LayeredV2ExecuteModelState):
            return self._sample_layered_v2(grammar_output, state)
        return self._sample_tokens_normal(grammar_output)

    def pool(self):
        output = super().pool()
        # Pooling models never go through sample_tokens() in the v2
        # architecture: the worker calls pool() directly, and the engine skips
        # sampling whenever execute_model()/pool() returns a non-None output.
        # Close the dump cycle here, mirroring v1 which finalizes after
        # _pool() on the pooling path.
        self._finalize_dump_data()
        return output

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return apply_dspark_resident_kv_specs(
            super().get_kv_cache_spec(),
            self.vllm_config,
            self.speculator,
            sparse_offload_enabled=self.ascend_config.sparse_kv_offload_config.enabled,
            is_last_pp_rank=self.is_last_pp_rank,
            shared_kv_cache_layers=getattr(self, "shared_kv_cache_layers", None),
        )

    def _init_kv_zero_meta(self) -> None:
        """Use a tuple-aware zeroer for Ascend V1-compatible cache protocols."""
        self.kv_block_zeroer = AscendV2KVBlockZeroer(
            self.device,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            static_forward_context=self.compilation_config.static_forward_context,
            num_blocks=self.kv_cache_config.num_blocks,
            cache_dtype=self.cache_config.cache_dtype,
        )
        if is_c8_mxfp_kv_quant(self.vllm_config):
            # The zeroer skips the static V scales (4th tuple component); the
            # wrapper restores them on pages reassigned from Mamba.
            self.kv_block_zeroer = C8KVBlockZeroer(  # type: ignore[assignment]
                self.kv_block_zeroer, self._c8_mxfp_v_scale_caches, fill_mxfp_v_scale_cache
            )

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        # Reinitialization can change the active attention backends.
        self._skip_spec_cpu_length_sync = False
        # Match V1's physical buffer plan without mutating the scheduler's
        # logical cache configuration. Allocation must honor zero-stride aliases.
        kv_cache_config = deepcopy(kv_cache_config)
        sparse_cfg = self.ascend_config.sparse_kv_offload_config
        configure_dspark_kv_transfer(
            self.vllm_config, self.speculator, kv_cache_config, is_last_pp_rank=self.is_last_pp_rank
        )
        resident_draft_names = get_resident_dspark_layer_names(
            self.vllm_config,
            self.speculator,
            sparse_offload_enabled=sparse_cfg.enabled,
            is_last_pp_rank=self.is_last_pp_rank,
            shared_kv_cache_layers=getattr(self, "shared_kv_cache_layers", None),
        )
        # P also needs persistent prompt draft KV until D acknowledges its
        # transfer. It must never alias target layerwise scratch buffers.
        persistent_draft_names = resident_draft_names | set(getattr(kv_cache_config, "dspark_draft_layer_names", ()))
        apply_layerwise_kv_cache_plan(kv_cache_config, self.vllm_config, excluded_layer_names=persistent_draft_names)
        if sparse_cfg.enabled:
            self.sparse_kv_offload_manager = init_sparse_kv_offload_manager(
                self.vllm_config, kv_cache_config, sparse_cfg
            )
            self.model_state._offload_live_req_ids = self.req_states.req_id_to_index
            self.model_state._offload_draft_layer_names = (
                getattr(self.speculator, "draft_attn_layer_names", set[str]()) - resident_draft_names
                if sparse_cfg.use_fused_copy_sfa
                else set()
            )
        with graph_manager_wrapper(self):
            super().initialize_kv_cache(
                kv_cache_config,
                kv_cache_allocation_context=kv_cache_allocation_context,
            )
            self._uses_flashmla = any(
                getattr(group.get_metadata_builder(0), "flashmla_state", None) is not None
                for groups in self.attn_groups
                for group in groups
            )
            if self.pcp_manager is not None:
                assert isinstance(self.pcp_manager, AscendPCPManager)
                self.pcp_manager.vllm_config = self.vllm_config
                self.pcp_manager.kv_cache_config = kv_cache_config
                self.pcp_manager.global_input_buffers = self.input_buffers
                self.model_state.pcp_manager = self.pcp_manager
                if self.speculator is not None:
                    self.speculator.pcp_manager = self.pcp_manager
        bind_dspark_context_receiver(
            self.vllm_config,
            sparse_offload_enabled=sparse_cfg.enabled,
            is_last_pp_rank=self.is_last_pp_rank,
            max_requests=self.max_num_reqs,
        )
        if sparse_cfg.enabled and self.speculator is not None:
            # Resident MLA DSpark has no host-pool LRU or LIM tail state.
            self.model_state._offload_draft_attn_groups = (
                [] if resident_draft_names else getattr(self.speculator, "attn_groups", [])
            )
        if any(is_circular_kv_cache_spec(group.kv_cache_spec) for group in self.kv_cache_config.kv_cache_groups):
            from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41Compressor

            for module in self.model.modules():
                if isinstance(module, DeepseekV41Compressor) and module.ratio == 2:
                    module.prepare_ring_compressor(self.max_num_tokens, self.device)
        prepare_v41_source_rope(self)
        # Recreate along with KV initialization: profiling capture owns a
        # throwaway model state and must not leak event/buffer bindings.
        self.model_state.device_metadata = (
            TargetDeviceMetadata()
            if uses_a5_packed_cache() and self.model_config.architecture == "DeepseekV41ForCausalLM"
            else None
        )

        # Upstream has bound every local cache; publish sealed plans before
        # the worker can warm up, capture graphs or execute prefill requests.
        from vllm_ascend.ops.kda_state_copy_plan import initialize_kda_state_copy

        initialize_kda_state_copy(
            self.vllm_config.compilation_config.static_forward_context,
            self.vllm_config.scheduler_config.max_num_seqs,
        )

        copy_mamba_state = getattr(self.model_state, "copy_state_after_kv_load", None)
        if (
            callable(copy_mamba_state)
            and has_kv_transfer_group()
            and getattr(get_kv_transfer_group(), "requires_mamba_state_copy_after_load", False)
        ):
            self.kv_connector = MambaCopyKVConnector(self.kv_connector, copy_mamba_state)

        # Only target-model layers determine whether FIA is in use. This flag
        # is used for adaptive verification handling.
        draft_layer_names: set[str] = getattr(self.speculator, "draft_attn_layer_names", set())
        self.use_fia = any(
            (group.backend is AscendAttentionBackend or group.backend is AscendMLABackend)
            and any(layer_name not in draft_layer_names for layer_name in group.layer_names)
            for groups in self.attn_groups
            for group in groups
        )

        # Legacy (pre-AuxOutput) R3 path; the getattr keeps MRv2 startable on a
        # vLLM lane where ModelConfig no longer exposes the flag.
        if getattr(self.model_config, "enable_return_routed_experts", False):
            self.init_routed_experts_capturer()

        self.kvpp = KVPPRuntime.create_from_kv_cache(
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
            static_forward_context=self.compilation_config.static_forward_context,
        )
        self.model_state.kvpp_runtime = self.kvpp
        self._skip_spec_cpu_length_sync = self._can_skip_spec_cpu_length_sync()
        logger.info("Speculative CPU length synchronization bypass enabled: %s", self._skip_spec_cpu_length_sync)

        self._fill_c8_mxfp_v_scale_caches()

        if getattr(self, "_layered_prefill_enabled", False):
            # Captured group graphs and private attention builders reference
            # KV addresses/specifications from this initialization only.
            self._layered_group_executor = LayeredPrefillGroupExecutor(self)
            self._layered_single_forward_executor = LayeredV2SingleForward(self, self._layered_group_executor)
            self._layered_pp_stage_executor = None
            self._layered_decode_buffers = LayeredV2Buffers()
            self.layered_prefill_state = LayeredPrefillStateStore()

    def _fill_c8_mxfp_v_scale_caches(self) -> None:
        """Initialize C8 V scales and retain views for page reassignment."""
        self._c8_mxfp_v_scale_caches = []
        num_blocks = self.kv_cache_config.num_blocks
        if num_blocks <= 0:
            return
        for layer in self.compilation_config.static_forward_context.values():
            if isinstance(getattr(layer, "impl", None), AscendC8MXFPAttentionBackendImpl):
                value_scale_cache = layer.kv_cache[3]
                assert value_scale_cache.shape[0] % num_blocks == 0
                kernel_blocks_per_page = value_scale_cache.shape[0] // num_blocks
                self._c8_mxfp_v_scale_caches.append((layer.v_cache_scale, value_scale_cache, kernel_blocks_per_page))
                fill_mxfp_v_scale_cache(layer.v_cache_scale, value_scale_cache)

    def shutdown(self) -> None:
        # Drop C8 packet views before the parent collects tensors and
        # empties the allocator cache.
        entries = getattr(self, "_c8_mxfp_v_scale_caches", None)
        if entries is not None:
            entries.clear()
        zeroer = getattr(self, "kv_block_zeroer", None)
        if isinstance(zeroer, C8KVBlockZeroer):
            zeroer.entries.clear()
            self.kv_block_zeroer = zeroer.delegate
        super().shutdown()

    def _register_sparse_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Bind host pools before the V2 KV connector registers its destinations."""
        manager = self.sparse_kv_offload_manager
        if manager is None:
            return
        manager.register_kv_caches(kv_caches)
        if not self.ascend_config.sparse_kv_offload_config.use_fused_copy_sfa:
            return

        from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl

        owners: dict[int, AscendSFAKVOffloadImpl] = {}
        for layer in self.compilation_config.static_forward_context.values():
            impl = getattr(layer, "impl", None)
            if not isinstance(impl, AscendSFAKVOffloadImpl):
                continue
            shared = impl.topk_indices_buffer
            if shared is None:
                continue
            key = shared.data_ptr()
            if impl.skip_topk:
                if key not in owners:
                    raise RuntimeError("fused_copy_sfa shared attention precedes its indexer owner")
                impl.lim_indexer_owner = owners[key]
            else:
                owners[key] = impl
        for layer_name in manager.offload_layer_names:
            layer = self.compilation_config.static_forward_context[layer_name]
            layer.impl.bind_copy_sfa_kv_cache(manager, layer_name)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
    ):
        if getattr(self, "_boundary_prefill_enabled", False) and not dummy_run and not is_profile:
            # A cancelled/preempted boundary can return directly to native
            # execution. Consume its cleanup on that step before the native
            # lifecycle removes those IDs from subsequent scheduler outputs.
            self.layered_prefill_state.clear_many(getattr(scheduler_output, "finished_req_ids", ()) or ())
            self.layered_prefill_state.clear_many(getattr(scheduler_output, "preempted_req_ids", ()) or ())
        layered_plan = getattr(scheduler_output, "layered_prefill_plan", None)
        if layered_plan is not None and not dummy_run and not self._layered_prefill_v2_ready:
            raise NotImplementedError("Layered Prefill V2 model adapter is not ready")
        self._cpp_execution_time_ms = None
        profiling_config = self.ascend_config.scheduler_config.profiling_chunk_config
        execution_start_time = _start_profiling_chunk_timing(
            profiling_config,
            scheduler_output,
        )

        # Preemption stores must complete before the parent updates states and
        # reuses or zeroes the preempted requests' physical KV cache blocks.
        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        dump_forward = dummy_run or scheduler_output.total_num_scheduled_tokens > 0
        # Eager PrecisionDebugger is started only for real scheduler steps.
        # Graph dumping is already active from load_model(), so dummy forwards
        # only need to flush the graph dumper below without writing a step.
        if dump_forward and not dummy_run:
            self._start_dump_data(scheduled_tokens=scheduler_output.num_scheduled_tokens)

        self.model_state.kvpp_is_dummy_run = dummy_run or is_profile
        dp_coordination_context = (
            skip_dp_coordination() if should_skip_allreduce_across_dp_group(self.vllm_config) else nullcontext()
        )
        forward_failed = True
        with dp_coordination_context:
            try:
                layered_plan = getattr(scheduler_output, "layered_prefill_plan", None)
                if layered_plan is not None and not dummy_run:
                    output = self._execute_layered_step(
                        scheduler_output,
                        layered_plan,
                        intermediate_tensors=intermediate_tensors,
                    )
                else:
                    output = super().execute_model(
                        scheduler_output,
                        intermediate_tensors=intermediate_tensors,
                        dummy_run=dummy_run,
                        skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                        is_profile=is_profile,
                        context_len=context_len,
                        valid_dummy_state_slots=valid_dummy_state_slots,
                    )
                forward_failed = False
            finally:
                finish_execution = getattr(self.model_state, "finish_execution", None)
                if finish_execution is not None:
                    finish_execution(failed=forward_failed)
        if not dummy_run and not is_profile:
            self._maybe_send_draft_kv(scheduler_output)
        self.model_state.kvpp_is_dummy_run = False
        if dummy_run and lmhead_tp_enable() and not is_profile and self.is_last_pp_rank:
            # lmhead TP: idle ranks never call sample(); join the target head
            # here at capacity, before _dummy_run replays the dummy propose.
            if self.execute_model_state is None:
                raise RuntimeError(
                    "lmhead TP dummy join expects execute_model_state published by the upstream dummy execute_model."
                )
            dummy_indices = torch.zeros(
                self._lmhead_tp_max_num_logits(),
                dtype=torch.int64,
                device=self.device,
            )
            self.model.compute_logits(self.execute_model_state.hidden_states[dummy_indices])
        self.kvpp.complete_forward()

        if dump_forward:
            if dummy_run:
                self._finalize_dump_data(dump=False)
            elif isinstance(output, IntermediateTensors) and self.is_pooling_model:
                # Non-last pooling ranks do not reach pool(). Decoding ranks
                # finalize after sample_tokens(), including non-last PP ranks.
                self._finalize_dump_data()

        self._cpp_execution_time_ms = _finish_profiling_chunk_timing(
            profiling_config,
            execution_start_time,
        )

        # TODO: Remove this workaround once upstream vLLM restores idle DP ranks'
        # logical request counts before draft DP synchronization.
        # In adaptive verification, idle DP ranks dummy-run at target graph capacity.
        # Restore their request counts before draft DP sync to avoid inflating all ranks.
        if (
            dummy_run
            and not is_profile
            and context_len == 0
            and self.adaptive_verification is not None
            and self.execute_model_state is not None
        ):
            state = self.execute_model_state
            self.execute_model_state = state._replace(
                input_batch=replace(state.input_batch, num_reqs=len(scheduler_output.num_scheduled_tokens))
            )
        return output

    def _maybe_send_draft_kv(self, scheduler_output: SchedulerOutput) -> None:
        """Write draft KV on P, then let D pull the exact allocated cache pages."""
        if not getattr(self, "pd_dspark_aux_layer_ids", ()) or not self.is_last_pp_rank:
            return
        connector, metadata = find_dspark_context_connector(
            get_kv_transfer_group(), scheduler_output.kv_connector_metadata
        )
        requests = getattr(metadata, "requests", {})
        if not any(getattr(req_meta, "dspark_context_generation", None) for req_meta in requests.values()):
            return
        state = self.execute_model_state
        if state is None or state.aux_hidden_states is None:
            if scheduler_output.total_num_scheduled_tokens:
                raise RuntimeError("P produced no target auxiliary states for an active remote DSpark request")
            return
        prefix_store = find_dspark_prefix_connector(get_kv_transfer_group(), scheduler_output.kv_connector_metadata)
        send_dspark_prefill_kv(
            self.speculator,
            state.input_batch,
            state.aux_hidden_states,
            requests,
            connector,
            self._dspark_prefill_progress,
            getattr(scheduler_output, "finished_req_ids", ()) or (),
            prefix_connector=prefix_store[0] if prefix_store is not None else None,
            prefix_metadata=prefix_store[1] if prefix_store is not None else None,
        )

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Override GPUModelRunner.profile_run for Ascend NPUs.
        When running moe models, we need an extra dummy run with mc2_tokens_capacity tokens to reserve
        necessary HCCL buffer for the MC2 operator before standard `profile_run`. Additionally, we set
        override_mrv2_in_profile_run to True to force moe load to be balanced when executing `profile_run`
        """
        sparse_cfg = self.ascend_config.sparse_kv_offload_config
        if sparse_cfg.enabled:
            allocate_kv_offload_topk_profile_buffers(self.get_kv_cache_spec(), self.vllm_config, sparse_cfg)
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        # The initial dummy profile bypasses PCP batch partitioning. Profile
        # rank-local tokens; the existing MoE preparation still handles TP.
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        global_max_num_tokens: int = self.max_num_tokens
        profile_max_num_tokens = (global_max_num_tokens + pcp_size - 1) // pcp_size
        with override_mrv2_in_profile_run(True):
            if (
                mc2_tokens_capacity is not None
                and profile_max_num_tokens > mc2_tokens_capacity
                and select_moe_comm_method(mc2_tokens_capacity, self.vllm_config)
                in {MoECommType.MC2, MoECommType.FUSED_MC2}
            ):
                # Use a call-scoped bypass because skip_compiled would require runner-specific ForwardContext plumbing.
                with disable_compilation(self.get_model()):
                    self._dummy_run(mc2_tokens_capacity, skip_attn=True, skip_eplb=True, is_profile=True)
            self.max_num_tokens: int = profile_max_num_tokens
            try:
                super().profile_run()
            finally:
                self.max_num_tokens = global_max_num_tokens
        self._profile_layered_frontier_budget()

    def gather_batch_req_state(self, scheduler_output: SchedulerOutput, dummy_run: bool):
        batch_state, uniform_token_count = super().gather_batch_req_state(scheduler_output, dummy_run)
        kv_cfg = self.vllm_config.kv_transfer_config
        is_pd_decode_node = kv_cfg is not None and kv_cfg.is_kv_consumer and not kv_cfg.is_kv_producer
        if batch_state is not None and is_pd_decode_node:
            pd_decode_recompute = (
                batch_state.is_prefilling_np
                & (batch_state.num_computed_prefill_tokens_np > 0)
                & (batch_state.num_scheduled_tokens == self.decode_query_len)
                & (
                    batch_state.num_computed_prefill_tokens_np + batch_state.num_scheduled_tokens
                    >= batch_state.prefill_len_np
                )
            )
            if np.any(pd_decode_recompute):
                batch_state.is_prefilling_np[pd_decode_recompute] = False
                batch_state = batch_state._replace(has_prefill=bool(batch_state.is_prefilling_np.any()))
                uniform_token_count = vllm_model_runner.get_uniform_decode_token_count(
                    len(batch_state.req_ids),
                    batch_state.num_tokens,
                    int(batch_state.num_scheduled_tokens.max()),
                    batch_state.has_prefill,
                )
        if self._uses_flashmla and batch_state is not None:
            # The backend order must precede derived inputs and graph dispatch.
            batch_state, uniform_token_count = prepare_flashmla_batch_req_state(batch_state, uniform_token_count)
        return batch_state, uniform_token_count

    def _check_finegrained_tp_graph_step(self, cg_mode: CUDAGraphMode) -> None:
        # Eager dispatch keeps per-rank token counts; the cross-DP o_proj/MLP exchanges would desync.
        if self._finegrained_tp_requires_graph and cg_mode == CUDAGraphMode.NONE:
            raise RuntimeError(
                "o_proj / MLP TP require every step on a captured graph: this step dispatched "
                "to eager, which desyncs the cross-DP HCCL collectives (mixed or oversized "
                "batch, a request-arrival step misclassified as prefill, or a full prefill "
                "scheduled locally — a request sent directly to the decode node)."
            )

    def prepare_inputs(  # type: ignore[misc]
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: BatchReqState,
        batch_desc: BatchExecutionDescriptor,
    ) -> AscendInputBatch:
        """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to prepare seq_lens_cpu here.
        """
        self._check_finegrained_tp_graph_step(batch_desc.cg_mode)
        num_tokens = batch_req_state.num_tokens
        num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
        graph_num_reqs = batch_desc.num_reqs
        global_graph_num_reqs = (
            self.pcp_manager.get_global_graph_num_reqs(batch_desc) if self.pcp_manager is not None else None
        )
        if global_graph_num_reqs is not None:
            graph_num_reqs = global_graph_num_reqs
            num_tokens_after_padding = global_graph_num_reqs * batch_desc.uniform_token_count
        assert num_tokens > 0

        req_ids = batch_req_state.req_ids

        self._update_seq_lens_cpu(scheduler_output, req_ids)

        num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
        idx_mapping_np = batch_req_state.idx_mapping_np
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)
        num_reqs = len(req_ids)

        num_valid_tokens = num_scheduled_tokens_np
        if scheduler_output.scheduled_spec_decode_tokens:
            num_valid_tokens = np.array(
                [
                    num_toks - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                    for num_toks, i in zip(num_scheduled_tokens_np, req_ids)
                ],
                dtype=np.int32,
            )
        attn_state = build_attn_state(
            self.vllm_config,
            self.input_buffers.seq_lens_np,
            num_reqs,
            num_scheduled_tokens_np,
            num_valid_tokens,
            kv_cache_config=self.kv_cache_config,
        )

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_draft_tokens_per_req = None
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        else:
            num_draft_tokens_per_req = np.fromiter(
                (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
            total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
            num_logits = num_draft_tokens_per_req + num_bonus_tokens
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

        adaptive_verification_manager = self.adaptive_verification
        adaptive_verification_active = (
            adaptive_verification_manager is not None and num_draft_tokens_per_req is not None
        )
        num_scheduled_tokens_upper_bound = num_scheduled_tokens_np
        if adaptive_verification_active:
            num_scheduled_tokens_np, cu_num_logits_np = adaptive_verification_manager.compact_batch(
                num_draft_tokens_per_req, num_scheduled_tokens_np, cu_num_logits_np
            )
        # Get query_start_loc.
        # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
        # See _pad_query_start_loc_for_fia.
        # Inputs are still global here; eager PCP descriptors carry local counts.
        num_reqs_padded = max(num_reqs, graph_num_reqs or 0)
        query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens

        if batch_desc.cg_mode == CUDAGraphMode.FULL and not adaptive_verification_manager:
            # This is only required for vllm-ascend.
            query_start_loc_np, num_reqs_padded = self._pad_query_start_loc_for_fia(
                num_tokens_after_padding,
                num_reqs_padded,
                num_reqs,
                query_start_loc_np,
                batch_desc.cg_mode,
                graph_num_reqs,
                uniform_query_len=batch_desc.uniform_token_count,
            )

        query_start_loc = self.input_buffers.query_start_loc
        async_copy_to_gpu(query_start_loc_np, out=query_start_loc)

        if adaptive_verification_active:
            cu_num_logits, query_start_loc, total_num_draft_tokens = adaptive_verification_manager.reallocate_drafts(
                req_ids, idx_mapping
            )
            total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens

            # Non-fia backends skip padding query boundary when using adaptive verification
            if self.use_fia:
                query_start_loc_np[: num_reqs + 1] = query_start_loc[: num_reqs + 1].cpu().numpy()
                query_start_loc_np[num_reqs + 1 :] = int(query_start_loc_np[num_reqs])

        if self.use_fia and adaptive_verification_manager:
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                query_start_loc_np, num_reqs_padded = self._pad_adaptive_query_start_loc_for_fia(
                    num_tokens_after_padding,
                    num_reqs_padded,
                    num_reqs,
                    query_start_loc_np,
                )

            query_start_loc = self.input_buffers.query_start_loc
            async_copy_to_gpu(query_start_loc_np, out=query_start_loc)

        if draft_tokens:
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, self.decode_query_len
            )

        query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
        query_start_loc = query_start_loc[: num_reqs_padded + 1]
        self.eplb.set_batch_phase(batch_req_state.has_prefill)

        # Graph dispatch may classify a PD prompt-tail step as decode, but
        # its input still comes from all_token_ids rather than sampled tokens.
        # Keep input preparation tied to the actual prefill progress so the
        # prompt tail and MTP lookahead are populated even on a decode graph.
        if np.any(batch_req_state.num_computed_prefill_tokens_np < batch_req_state.prefill_len_np):
            prepare_prefill_inputs(
                self.input_buffers.input_ids,
                self.req_states.next_prefill_tokens,
                idx_mapping,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]
        if adaptive_verification_active and self.use_fia:
            self.input_buffers.seq_lens_np[:num_reqs] = seq_lens[:num_reqs].cpu().numpy()

        # Pad for full CUDA graph mode.
        self.input_buffers.seq_lens_np[num_reqs:] = 0

        dcp_local_seq_lens = None
        # Main computes DCP lengths in the inherited execute_model after PCP
        # partitioning (vLLM #55212).

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            self.model_state.num_new_sampled_tokens_per_step,
        )

        # CPU upper bound on seq_lens (num_computed_tokens + num_scheduled_tokens).
        # Added by vLLM PR #40654 to avoid GPU->CPU sync for seq_lens.
        num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]
        seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
        np.add(
            num_computed_tokens_np,
            num_scheduled_tokens_upper_bound,
            out=seq_lens_cpu_upper_bound_np[:num_reqs],
        )
        seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

        prompt_lens = None
        if self.model_config.rswa_window is not None:
            # prompt_lens is only used in R-SWA case.
            prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

        input_batch = AscendInputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs_padded,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens_upper_bound,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            num_computed_tokens_np=num_computed_tokens_np,
            prefill_len_np=batch_req_state.prefill_len_np,
            num_computed_prefill_tokens_np=batch_req_state.num_computed_prefill_tokens_np,
            is_prefilling_np=batch_req_state.is_prefilling_np,
            has_prefill=batch_req_state.has_prefill,
            input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
            positions=self.input_buffers.positions[:num_tokens_after_padding],
            is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            # TODO: only populated for R-SWA (not supported yet).
            prompt_lens=prompt_lens,
            # extra attributes for ascend npus.
            seq_lens_np=self.input_buffers.seq_lens_np,
            attn_state=attn_state,
        )
        # vLLM main (#53867) changed maybe_partition_pcp_batch to take the
        # whole batch descriptor instead of padded_num_tokens.
        input_batch = vllm_model_runner.pcp.maybe_partition_pcp_batch(
            self.pcp_manager,
            input_batch,
            batch_desc=batch_desc,
        )

        # For mla/sfa, update cos/sin. Here is for execute_model.
        update_cos_sin(input_batch.positions)

        return input_batch

    def prepare_dummy_attn(
        self, input_batch: AscendInputBatch, valid_state_slots: bool = False
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables, slot_mappings = super().prepare_dummy_attn(
            input_batch,
            valid_state_slots=valid_state_slots,
        )
        prepare_v41_dummy_ring_state(self, input_batch.num_reqs)
        slot_mappings = self._maybe_extend_slot_mappings(input_batch, slot_mappings)
        return block_tables, slot_mappings

    def _maybe_extend_slot_mappings(self, input_batch: AscendInputBatch, slot_mappings: torch.Tensor) -> torch.Tensor:
        if input_batch.num_tokens_after_padding > input_batch.num_tokens:
            # The parent already filled the entire persistent slot buffer.
            slot_mappings = self.block_tables.slot_mappings[:, : input_batch.num_tokens_after_padding]
        return slot_mappings

    def _lmhead_tp_max_num_logits(self) -> int:
        """Logits row capacity every rank agrees on (config-derived:
        ``max_num_reqs * decode_query_len``); drift desyncs and hangs."""
        return lmhead_tp_max_num_logits(self.max_num_reqs, self.decode_query_len)

    def sample(self, hidden_states, input_batch, grammar_output):
        """Override GPUModelRunner.sample for lmhead TP: every rank must feed
        compute_logits the same row count — pad up to ``_lmhead_tp_max_num_logits()``
        and trim back; prompt_logprobs stays unsupported (same as V1)."""
        if not lmhead_tp_enable():
            return super().sample(hidden_states, input_batch, grammar_output)

        num_logits = input_batch.logits_indices.shape[0]
        capacity = self._lmhead_tp_max_num_logits()
        # V1-style index pad: pad a private copy of the indices (input_batch keeps
        # the real ones; the V2 sampler gathers penalties by them) so one gather
        # feeds compute_logits the capacity rows; zero entries gather row 0.
        sample_indices = lmhead_tp_pad_rows(
            input_batch.logits_indices,
            capacity,
            "max_num_reqs * decode_query_len",
        )
        logits = self.model.compute_logits(hidden_states[sample_indices])
        logits = logits[:num_logits]

        # Dispatch tail mirrors GPUModelRunner.sample; refresh it on main bumps.
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected

    @contextmanager
    def _cap_parallel_draft_dummy_reqs(self, uniform_decode: bool):
        # Profiling can exceed the speculator's query buffer.
        original_max_num_reqs = self.max_num_reqs
        if self.speculator is not None and not uniform_decode:
            # Other speculators use one query row per request in vLLM v0.30.0.
            query_width = getattr(self.speculator, "num_query_per_req", 1)
            self.max_num_reqs = min(original_max_num_reqs, self.max_num_tokens // query_width)
        try:
            yield
        finally:
            self.max_num_reqs = original_max_num_reqs

    @contextmanager
    def _preserve_dummy_query_tokens(self, num_tokens: int, uniform_decode: bool):
        dummy_query_tokens: int | None = None
        if (
            # FULL modes can also pad requests; only change token-only PIECEWISE padding.
            self.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
            # PCP prepares and partitions its own dummy input layout.
            and self.pcp_manager is None
            # Hybrid models also need recurrent-state metadata to stay aligned.
            # TODO: Verify whether this hybrid-model guard can be removed.
            and not self.model_config.is_hybrid
        ):
            # Match the logical token count in the upstream dummy scheduler.
            dummy_query_tokens = max(num_tokens, self.decode_query_len) if uniform_decode else num_tokens

        previous_dummy_tokens = self.input_buffers.dummy_num_tokens
        self.input_buffers.dummy_num_tokens = dummy_query_tokens
        try:
            yield
        finally:
            self.input_buffers.dummy_num_tokens = previous_dummy_tokens

    @contextmanager
    def _cpp_profile_dummy(self, enabled: bool):
        """Scope the target-only prefill semantics used by CPP profiling."""
        if not enabled:
            yield
            return

        previous_profile_cpp = getattr(self.input_buffers, "profile_cpp", False)
        previous_speculator = self.speculator
        previous_max_num_reqs = self.max_num_reqs
        try:
            self.input_buffers.profile_cpp = True
            # CPP profiles the target prefill. This also matches MRV1, where the
            # drafter dummy run is explicitly skipped for profile_cpp.
            self.speculator = None
            self.max_num_reqs = 1
            yield
        finally:
            self.input_buffers.profile_cpp = previous_profile_cpp
            self.speculator = previous_speculator
            self.max_num_reqs = previous_max_num_reqs

    @step_eplb_after(is_dummy=True)
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = False,
        uniform_decode: bool = False,
        context_len: int = 0,
        skip_eplb: bool = False,
        is_profile: bool = False,
        profile_cpp: bool = False,
        **kwargs,
    ):
        """Balanced dummy routing for adaptive-verification profiling; EPLB
        steps via ``step_eplb_after`` (#17233). The lmhead TP join lives in the
        ``execute_model`` tail — joining after ``super()._dummy_run`` deadlocks."""
        skip_ring = bool(kwargs.pop("skip_gdn_state_update", False))
        # Adaptive verification profiles eager tail sizes after graph capture.
        # Use balanced dummy routing, as the initial memory profile does, so a
        # synthetic router hotspot cannot exhaust one EP rank during startup.
        profile_adaptive_tail = self.adaptive_verification is not None and context_len > 0
        if profile_adaptive_tail and self.ascend_config.xlite_graph_config.enabled:
            logger.warning_once(
                "Adaptive verification cost profiling with XLite enabled may "
                "produce inaccurate costs because balanced MoE profiling sets "
                "the profile-run marker, which makes XLite bypass its graph path."
            )
        load_balance_ctx = override_mrv2_in_profile_run(True) if profile_adaptive_tail else nullcontext()
        with (
            # Disable the CPP drafter before entering contexts that inspect it.
            self._cpp_profile_dummy(profile_cpp),
            # Preserve scheduled query tokens before PIECEWISE graph padding.
            self._preserve_dummy_query_tokens(num_tokens, uniform_decode),
            # TODO: Remove this context and its use after the next main2main
            # includes https://github.com/vllm-project/vllm/pull/56448.
            self._cap_parallel_draft_dummy_reqs(uniform_decode),
            skip_ring_state_update(skip_ring),
            load_balance_ctx,
        ):
            output = super()._dummy_run(
                num_tokens,
                *args,
                skip_attn=skip_attn,
                uniform_decode=uniform_decode,
                context_len=context_len,
                skip_eplb=True,
                is_profile=is_profile,
                **kwargs,
            )
        # Flush the dump window opened across dummy forwards without writing
        # data, so capture/profiling forwards do not leak into the first real
        # step (v1 parity: v1 finalizes with dump=False at the end of
        # _dummy_run).
        self._finalize_dump_data(dump=False)
        return output

    def _can_skip_spec_cpu_length_sync(self) -> bool:
        """Use device lengths for K3/DSpark FlashMLA, including DCP."""
        parallel = self.vllm_config.parallel_config
        if (
            self.model_config.architecture
            not in {"KimiLinearForCausalLM", "KimiK3ForCausalLM", "KimiK3ForConditionalGeneration"}
            or not getattr(self.speculator, "_draft_mla_uses_device_lengths", False)
            # PP/PCP and offload paths have not been audited for host readers.
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or self.adaptive_verification is not None
            or self.ascend_config.sparse_kv_offload_config.enabled
            or self.ascend_config.kvpp_config.size != 1
        ):
            return False

        # FlashMLA derives DCP-local lengths from rejection-corrected device
        # counters. K3's non-MLA attention uses the KDA device-length path.
        return all_mla_use_flashmla(self.model)

    def postprocess_sampled(
        self,
        idx_mapping,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc=None,
    ):
        """Refresh rejection-corrected host counts when a consumer needs them."""
        if (
            self.use_pp
            and not self.is_last_pp_rank
            and isinstance(self.model_state, MambaHybridModelState)
            and self.cache_config.mamba_cache_mode == "align"
        ):
            # Deferred PP results belong to an older batch than input_block_tables.
            # Restore its rows before Mamba aligns the accepted recurrent state.
            # Postprocess skips -1 (freed/unsampled) rows; gather needs valid indices.
            self.block_tables.gather_block_tables(idx_mapping.clamp_min(0), idx_mapping.shape[0])
        super().postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
        )

        # The copy helper skips D2H only when both target and draft
        # backends are on the audited device-length path.
        # Non-last PP stages receive rejections without owning a speculator.
        if (
            self.speculator is not None and not is_deepseek_v41(self.model_config.hf_config)
        ) or self.sync_spec_pp_cpu_counts:
            self._copy_num_computed_tokens_to_cpu()

    def postprocess_num_computed_tokens(self, input_batch: AscendInputBatch) -> None:
        super().postprocess_num_computed_tokens(input_batch)
        # Unsampled prefill chunks must also refresh the next step's snapshot.
        if self.sync_spec_pp_cpu_counts:
            self._copy_num_computed_tokens_to_cpu()

    def _copy_num_computed_tokens_to_cpu(self):
        if self._skip_spec_cpu_length_sync:
            return
        # Attention metadata still needs exact CPU lengths. This non-blocking
        # D2H is waited on in _update_seq_lens_cpu, introducing a host/device
        # sync point that can break asynchronous scheduling overlap.
        default_stream = torch.cuda.current_stream()
        assert self.num_computed_tokens_stream is not None
        assert self.num_computed_tokens_cpu is not None
        with torch.npu.stream(self.num_computed_tokens_stream):
            self.num_computed_tokens_stream.wait_stream(default_stream)
            self.num_computed_tokens_cpu.copy_(
                self.req_states.num_computed_tokens.gpu,
                non_blocking=True,
            )
            self.num_computed_tokens_event.record()

    def _update_seq_lens_cpu(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: list[str],
    ):
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        # Keep the exact host mirror for consumers such as FIA and PCP.
        # Device-length backends retain the scheduler's CPU upper bound.
        if not self._skip_spec_cpu_length_sync and (
            (self.speculator is not None and not is_deepseek_v41(self.model_config.hf_config))
            or self.sync_spec_pp_cpu_counts
        ):
            # Blocks CPU submission until D2H completes; may stall the async pipeline.
            self.num_computed_tokens_event.synchronize()
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.num_computed_tokens_cpu[req_index]

        # Without a CPU consumer, retain the upstream optimistic upper bound.
        # prepare_pos_seq_lens still reads exact rejection-corrected NPU state.
        for i, req_id in enumerate(req_ids):  # type: ignore
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_cpu[req_index]
            self.input_buffers.seq_lens_cpu[i] = num_computed_tokens + num_scheduled_tokens[req_id]
        self.input_buffers.seq_lens_cpu[len(req_ids) :] = 0

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        query_start_loc_np: np.ndarray,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
        uniform_query_len: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        descriptor_num_reqs = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs_padded
        query_len = uniform_query_len or self.decode_query_len
        # This checks query lengths, not request phase: short prefills can also
        # match. Graph dispatch is responsible for excluding incompatible prefills.
        has_uniform_decode_query_lens = np.all(np.diff(query_start_loc_np[: num_reqs + 1]) == query_len)
        matches_uniform_decode_graph_shape = (
            has_uniform_decode_query_lens and num_tokens_padded == descriptor_num_reqs * query_len
        )
        if (
            cudagraph_runtime_mode == CUDAGraphMode.FULL
            and self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
            and not matches_uniform_decode_graph_shape
        ):
            num_reqs_padded = num_reqs
        else:
            # Preserve the captured request shape for uniform decode graphs.
            # GDN full graphs capture metadata at request granularity, so
            # collapsing all padded tokens into one request changes the graph
            # topology between capture and replay.
            num_reqs_padded = descriptor_num_reqs

        if has_uniform_decode_query_lens and num_tokens_padded == num_reqs_padded * query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        return query_start_loc_np, num_reqs_padded

    def _start_dump_data(self, **kwargs) -> None:
        if self.debugger is None or self._debugger_started:
            return
        self.debugger.start(self.model, **kwargs)
        self._debugger_started = True

    def _finalize_dump_data(self, **kwargs) -> None:
        if self.debugger is None or not self._debugger_started:
            return
        if hasattr(self.debugger, "stop"):
            self.debugger.stop()
            self._debugger_started = False

        self.debugger.step(**kwargs)

    def _pad_adaptive_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        query_start_loc_np: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Pad adaptive query boundary to the captured FULL graph request shape."""
        last_loc = int(query_start_loc_np[num_reqs])
        num_padding_tokens = num_tokens_padded - last_loc
        num_padding_reqs = num_reqs_padded - num_reqs
        assert num_padding_tokens >= 0 and num_padding_reqs >= 0

        if num_padding_reqs == 0:
            if num_padding_tokens > 0:
                query_start_loc_np[num_reqs + 1] = num_tokens_padded
                num_reqs_padded += 1
            return query_start_loc_np, num_reqs_padded

        cumulative_padding = np.arange(1, num_padding_reqs + 1, dtype=np.int32) * num_padding_tokens // num_padding_reqs
        query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = last_loc + cumulative_padding
        return query_start_loc_np, num_reqs_padded

    @contextmanager
    def _layered_p_buffers(self):
        """Swap in the P-only AscendInputBuffers for the Prefill sub-batch.

        Must not fork prepare_inputs: that path hard-codes self.input_buffers.
        D always uses the main buffers; P runs under this context (eager only).
        """
        if self._layered_input_buffers is None:
            raise RuntimeError(
                "Layered Prefill P buffers were not allocated; "
                "enable layered_prefill_config before constructing NPUModelRunner"
            )
        main = self.input_buffers
        self.input_buffers = self._layered_input_buffers
        try:
            yield
        finally:
            self.input_buffers = main

    @contextmanager
    def _layered_pp_handler_capture(self):
        """Defer PPHandler so D and final-P share one sampled-token slot."""
        handler = self.pp_handler
        if handler is None:
            yield None
            return
        capture = LayeredPPHandlerCapture(handler)
        self.pp_handler = capture
        try:
            yield capture
        finally:
            self.pp_handler = handler

    def update_pp_decode_requests(self):
        # Layered D still calls super().execute_model, which would consume the
        # one-slot queue a second time.  Lifecycle already applied it first.
        if self._layered_skip_pp_decode_update:
            return
        super().update_pp_decode_requests()

    def _execute_layered_step(
        self,
        scheduler_output: SchedulerOutput,
        layered_plan,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        """D then P dual sub-batch orchestration (V2, including PP>1).

        Default execution order D→P is a correctness constraint: Ascend stores
        model_state.attn_metadata as instance state consumed by
        ModelAclGraphManager.run_fullgraph (see migration plan §3.1).

        PP>1 reuses the V1 outer-worker protocol: one recv/send per scheduler
        step, with D/P rows packed into a single IntermediateTensors payload
        plus layered_pp_{d,p}_rows metadata.  Sampled tokens share one
        PPHandler slot (D ∪ final P); intermediate P groups are excluded.
        """
        if self.layered_prefill_model_adapter is None:
            raise RuntimeError("The loaded model does not have a layered prefill adapter")
        if self.pp_handler is not None:
            self.pp_handler.bind_plan(getattr(scheduler_output, "ring_step_plan", None))

        use_pp = self.parallel_config.pipeline_parallel_size > 1
        self._layered_compute_num_groups = int(layered_plan.num_groups)
        if not use_pp and intermediate_tensors is not None:
            raise RuntimeError("Layered Prefill V2 received PP intermediate tensors at PP=1")

        finished = getattr(scheduler_output, "finished_req_ids", ()) or ()
        preempted = getattr(scheduler_output, "preempted_req_ids", None) or ()
        self.layered_prefill_state.clear_many(finished)
        if preempted:
            self.layered_prefill_state.clear_many(preempted)

        d_req_ids, p_req_ids = split_d_p_req_ids(scheduler_output, layered_plan)
        single_forward = (
            bool(getattr(self, "_single_forward", False))
            and bool(p_req_ids)
            and len(p_req_ids) == 1
            and LayeredV2SingleForward.supports_plan(layered_plan)
            and LayeredV2SingleForward.supported(self)
        )
        pp_stage_mixed = LayeredV2PPStageForward.supported(self, layered_plan, d_req_ids, scheduler_output)
        # A stage with no work in this layer group relays it instead of
        # running the runner machinery.  Decided before the activation is
        # unpacked, so a stage ahead of the owner does not wait for a payload
        # it will never read.  The D/P-split path is excluded: it carries two
        # payloads, and relaying one of them needs its own reasoning.
        relay_action = None
        if use_pp and not d_req_ids:
            relay_action = self._layered_relay_stage_action(layered_plan)

        d_intermediate = None
        p_intermediate = None
        if use_pp and not self.is_first_pp_rank and relay_action is None:
            d_intermediate, p_intermediate = self._split_layered_pp_intermediate(intermediate_tensors)
        self._record_layered_step(
            layered_plan,
            n_d=len(d_req_ids),
            n_p=len(p_req_ids),
        )
        if use_pp and not self.is_first_pp_rank and d_req_ids and d_intermediate is None:
            raise RuntimeError("Layered PP D sub-batch did not receive activation")

        # One scheduler step → one lifecycle pass.  Must cover *all* new/cached
        # rows (including the P request) before either sub-batch forward.
        # Consume T-pp_size sampled tokens *before* finish/add so the generation
        # counters still match the PendingRecv snapshot; D's super().execute_model
        # then skips the same call via _layered_skip_pp_decode_update.
        self._layered_skip_pp_decode_update = False
        self._apply_scheduler_lifecycle(scheduler_output)
        self._layered_skip_pp_decode_update = True
        try:
            # Ascend builds seq_lens from num_computed_tokens_cpu.  That mirror
            # is normally refreshed in prepare_inputs only for
            # scheduled_cached_reqs rows; D/P sub-batches strip those fields
            # to avoid double update_requests, so sync explicitly here.
            scheduled_ids = list(scheduler_output.num_scheduled_tokens)
            self._sync_ascend_num_computed_tokens_cpu(scheduled_ids)
            self._sync_gpu_num_computed_tokens(scheduled_ids)

            if single_forward:
                return self._execute_layered_single_forward(scheduler_output, layered_plan, p_req_ids)

            if pp_stage_mixed:
                if self._layered_pp_stage_executor is None:
                    self._layered_pp_stage_executor = LayeredV2PPStageForward(self)
                return self._layered_pp_stage_executor.execute(
                    scheduler_output,
                    layered_plan,
                    d_req_ids,
                    d_intermediate,
                    p_intermediate,
                )

            if relay_action is not None:
                return self._execute_layered_relay_step(
                    scheduler_output,
                    relay_action,
                    intermediate_tensors,
                )

            d_state: ExecuteModelState | None = None
            pp_intermediates: list[IntermediateTensors] = []
            # D before P: Ascend stores attn_metadata on model_state for fullgraph.
            if d_req_ids:
                d_output = subset_scheduler_output(
                    scheduler_output,
                    d_req_ids,
                    layered_plan=None,
                    include_one_time_updates=False,
                )
                # Lifecycle already applied; strip rows that would re-enter it.
                d_output = self._strip_lifecycle_fields(d_output)
                d_failed = True
                try:
                    result = super().execute_model(d_output, intermediate_tensors=d_intermediate)
                    d_failed = False
                finally:
                    # Complete D metadata before P reuses the builder resources.
                    self.model_state.finish_execution(failed=d_failed)
                self.kvpp.complete_forward()
                d_state = self.execute_model_state
                self.execute_model_state = None
                if d_state is None:
                    raise RuntimeError("Layered Prefill V2 D sub-batch missing execute state")
                if use_pp and not self.is_last_pp_rank:
                    if not isinstance(result, IntermediateTensors):
                        raise RuntimeError(
                            "Layered Prefill V2 D sub-batch must return IntermediateTensors on a non-last PP rank"
                        )
                    if p_req_ids:
                        self._synchronize_layered_compute()
                        result = self._clone_intermediate_tensors(result)
                    pp_intermediates.append(result)
                elif result is not None:
                    raise RuntimeError("Layered Prefill V2 D sub-batch returned an unexpected output")
                if p_req_ids:
                    # P reuses model activation scratch; keep D logits inputs alive.
                    if getattr(self, "use_async_scheduling", False) or getattr(
                        getattr(self, "scheduler_config", None), "async_scheduling", False
                    ):
                        # Preserve per-step ownership outside the synchronous
                        # execution envelope validated for scratch reuse.
                        d_state = detach_execute_model_state(d_state)
                    else:
                        if not hasattr(self, "_layered_decode_buffers"):
                            self._layered_decode_buffers = LayeredV2Buffers()
                        d_state = self._layered_decode_buffers.detach_state(d_state)
                    # As in MRV1, different MoE backends may use different
                    # runtime streams. Keep this cross-subbatch dependency.
                    self._synchronize_layered_compute()

            p_state: ExecuteModelState | None = None
            p_layered_output: LayeredForwardOutput | None = None
            if p_req_ids:
                p_output = subset_scheduler_output(
                    scheduler_output,
                    p_req_ids,
                    layered_plan=layered_plan,
                    include_one_time_updates=False,
                )
                prefill_trace = nullcontext()
                profiler_config = getattr(self.vllm_config, "profiler_config", None)
                if getattr(profiler_config, "profiler", None) is not None:
                    ring_plan = getattr(scheduler_output, "ring_step_plan", None)
                    label = (
                        f"layered_pp_prefill/step={getattr(ring_plan, 'step_id', -1)}"
                        f"/group={layered_plan.group_id}"
                        f"/layers={layered_plan.group_start}:{layered_plan.group_end}"
                        f"/P={','.join(p_req_ids)}"
                    )
                    prefill_trace = torch.profiler.record_function(label)
                with self._layered_p_buffers(), prefill_trace:
                    p_state, p_layered_output = self._run_layered_prefill_subbatch(
                        p_output,
                        layered_plan,
                        p_intermediate,
                        moe_comm_token_count=scheduler_output.total_num_scheduled_tokens,
                    )
                    self.kv_connector.finish_forward()
                # Prompt KV writes may use auxiliary attention streams; the
                # next scheduler step can immediately replay a Decode graph.
                self._synchronize_layered_compute()
                if use_pp and not self.is_last_pp_rank:
                    assert p_layered_output is not None
                    pp_intermediates.append(
                        self.layered_prefill_model_adapter.to_intermediate_tensors(p_layered_output)
                    )

            self.execute_model_state = LayeredV2ExecuteModelState(
                scheduler_output=scheduler_output,
                d_state=d_state,
                p_state=p_state,
                sample_p=bool(getattr(layered_plan, "is_sampling_step", layered_plan.is_final_group)),
            )
            if use_pp and not self.is_last_pp_rank:
                d_pp = pp_intermediates[0] if d_req_ids else None
                p_pp = pp_intermediates[-1] if p_req_ids else None
                if d_pp is None and p_pp is None:
                    raise RuntimeError("Layered Prefill V2 non-last PP rank produced no intermediates")
                p_pp = self._layered_nonfused_pp_payload(layered_plan, p_pp)
                packed = self._combine_layered_pp_intermediate(d_pp, p_pp)
                packed.kv_connector_output = getattr(d_pp, "kv_connector_output", None) if d_pp is not None else None
                if packed.kv_connector_output is None and p_pp is not None:
                    packed.kv_connector_output = getattr(p_pp, "kv_connector_output", None)
                return packed
            return None
        finally:
            self._layered_skip_pp_decode_update = False

    def _synchronize_layered_compute(self) -> None:
        if self.parallel_config.pipeline_parallel_size <= 1:
            torch.npu.synchronize()
            return
        # DSA attention and MoE join auxiliary work onto the caller stream.
        # Include graph-parameter updates before reusing the D/P scratch, but
        # leave PPHandler's sampled-token receive pending until its queue slot
        # is consumed. A device-wide sync would drain that receive early.
        compute_stream = torch.npu.current_stream()
        update_stream = getattr(self, "update_stream", None)
        if update_stream is not None:
            compute_stream.wait_stream(update_stream)
        if not self._layered_async_compute_enabled():
            compute_stream.synchronize()

    def _layered_async_compute_enabled(self) -> bool:
        # Unsampled P retires through a completion event, including empty
        # local layer ranges. The separate relay bypass has no such event.
        if not envs_ascend.VLLM_ASCEND_LAYERED_ASYNC_COMPUTE or self.parallel_config.pipeline_parallel_size != 2:
            return False
        groups = getattr(self, "_layered_compute_num_groups", 0)
        if (
            not 2 <= groups <= self._num_hidden_layers()
            or getattr(self, "_skip_relay_stages", False)
            or getattr(self, "_fuse_mixed_batch", False)
            or getattr(self, "_same_layer_batch", False)
            or getattr(self, "use_async_scheduling", False)
            or getattr(getattr(self, "scheduler_config", None), "async_scheduling", False)
            or getattr(self, "dp_size", 1) != 1
            or getattr(getattr(self, "vllm_config", None), "kv_transfer_config", None) is not None
        ):
            return False
        from vllm_ascend.patch.worker.patch_v2.patch_uva import is_uva_available

        # Real UVA lets kernels read mutable CPU memory directly; device-side
        # ordering alone cannot protect the next CPU write to that memory.
        return not is_uva_available()

    def _complete_layered_output(self, output, state):
        if (
            self._layered_async_compute_enabled()
            and self.is_last_pp_rank
            and state.d_state is None
            and not state.sample_p
        ):
            # D/final-P already drains AsyncOutput in _sample_layered_v2.
            # Unsampled P must provide the same EngineCore completion boundary
            # or its empty result would retire KV/pinned-buffer ownership early.
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
            return LayeredPrefillAsyncOutput(output, event, keepalive=state)
        return output

    def _execute_layered_single_forward(self, scheduler_output, plan, p_req_ids):
        prepared = self._prepare_layered_batch(self._strip_lifecycle_fields(scheduler_output))
        self.kv_connector.pre_forward(scheduler_output)
        hidden, decode = self._layered_single_forward_executor.forward(prepared, plan)
        self.kv_connector.finish_forward()
        counters = self._layered_counters()
        counters["single_forward_steps"] = counters.get("single_forward_steps", 0) + 1
        if plan.is_sampling_step:
            d_state = None
            p_state = ExecuteModelState(
                input_batch=prepared.input_batch,
                attn_metadata=prepared.attn_metadata,
                slot_mappings_by_layer=prepared.slot_mappings_by_layer,
                hidden_states=hidden,
                aux_hidden_states=None,
                finished_req_ids=scheduler_output.finished_req_ids,
                dp_sync=getattr(prepared, "dp_sync", None),
                ec_connector_output=None,
                routed_experts=None,
                cudagraph_stats=None,
            )
        else:
            d_state = (
                None
                if decode is None
                else ExecuteModelState(
                    input_batch=decode.input_batch,
                    attn_metadata=decode.attn_metadata,
                    slot_mappings_by_layer=decode.slot_mappings_by_layer,
                    hidden_states=hidden,
                    aux_hidden_states=None,
                    finished_req_ids=scheduler_output.finished_req_ids,
                    dp_sync=getattr(prepared, "dp_sync", None),
                    ec_connector_output=None,
                    routed_experts=None,
                    cudagraph_stats=None,
                )
            )
            p_state = ExecuteModelState(
                input_batch=SimpleNamespace(req_ids=list(p_req_ids), num_reqs=len(p_req_ids)),
                attn_metadata=None,
                slot_mappings_by_layer=None,
                hidden_states=None,
                aux_hidden_states=None,
                finished_req_ids=scheduler_output.finished_req_ids,
                dp_sync=None,
                ec_connector_output=None,
                routed_experts=None,
                cudagraph_stats=None,
            )
        self.execute_model_state = LayeredV2ExecuteModelState(
            scheduler_output=scheduler_output,
            d_state=d_state,
            p_state=p_state,
            sample_p=bool(plan.is_sampling_step),
            prefill_req_ids=tuple(p_req_ids),
            single_forward=True,
        )
        return None

    def _num_hidden_layers(self) -> int:
        cfg = self.model_config
        hf = getattr(cfg, "hf_text_config", None) or getattr(cfg, "hf_config", None)
        value = getattr(hf, "num_hidden_layers", None)
        if value is None:
            value = getattr(cfg, "num_hidden_layers", None)
        if value is None:
            raise RuntimeError("Layered prefill needs model num_hidden_layers")
        return int(value)

    @staticmethod
    def _strip_lifecycle_fields(scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """Clear one-shot lifecycle fields after they have already been applied."""
        from dataclasses import replace

        from vllm.v1.core.sched.output import CachedRequestData

        return replace(
            scheduler_output,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            block_table_updates=None,
            finished_req_ids=set(),
            preempted_req_ids=set(),
            free_encoder_mm_hashes=[],
            scheduled_encoder_input_stats=None,
            new_block_ids_to_zero=None,
            kv_cache_block_copies=None,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            ec_manager_metadata=None,
        )

    def reset_layered_prefill_counters(self) -> None:
        self.layered_prefill_counters = empty_layered_prefill_counters()

    def _layered_counters(self) -> dict:
        counters = getattr(self, "layered_prefill_counters", None)
        if counters is None:
            counters = empty_layered_prefill_counters()
            self.layered_prefill_counters = counters
        return counters

    def _record_layered_step(
        self,
        layered_plan,
        *,
        n_d: int,
        n_p: int,
    ) -> None:
        counters = self._layered_counters()
        counters["execute_steps"] += 1
        counters["groups"].append(
            {
                "group_id": int(layered_plan.group_id),
                "num_groups": int(layered_plan.num_groups),
                "group_start": int(layered_plan.group_start),
                "group_end": int(layered_plan.group_end),
                "is_final": bool(layered_plan.is_final_group),
                "is_sampling_step": bool(getattr(layered_plan, "is_sampling_step", layered_plan.is_final_group)),
                "is_final_chunk": bool(getattr(layered_plan, "is_final_chunk", True)),
                "n_d": int(n_d),
                "n_p": int(n_p),
                "cached_tokens": int(
                    (getattr(layered_plan, "cached_tokens", None) or {}).get(layered_plan.prefill_req_ids[0], 0)
                    if getattr(layered_plan, "prefill_req_ids", None)
                    else 0
                ),
            }
        )

    def _record_layered_pp_slot(self, slot: dict | None) -> None:
        if not slot:
            return
        self._layered_counters()["pp_slots"].append(slot)

    def _record_layered_activation(self, layered_plan, source: str) -> None:
        counters = self._layered_counters()
        pp = get_pp_group()
        counters["activation_sources"].append(
            {
                "group_id": int(layered_plan.group_id),
                "pp_rank": int(pp.rank_in_group),
                "owner": int(self._layered_pp_group_owner(layered_plan)),
                "source": source,
            }
        )
        if source == "transport_frontier":
            counters["transport_frontier_steps"] += 1

    def layered_prefill_snapshot(self) -> dict:
        """JSON-safe worker state for smoke probes (via collective_rpc)."""
        parallel = getattr(self, "parallel_config", None)
        counters = self._layered_counters()
        group_executor = getattr(self, "_layered_group_executor", None)
        decode_graph_cache = getattr(group_executor, "decode_graph_cache", None)
        try:
            pp_rank = int(get_pp_group().rank_in_group)
        except Exception:
            pp_rank = -1
        return {
            "runner_class": f"{type(self).__module__}.{type(self).__name__}",
            "pp_size": int(getattr(parallel, "pipeline_parallel_size", -1)),
            "tp_size": int(getattr(parallel, "tensor_parallel_size", -1)),
            "pp_rank": pp_rank,
            "is_first_pp_rank": bool(getattr(self, "is_first_pp_rank", False)),
            "is_last_pp_rank": bool(getattr(self, "is_last_pp_rank", False)),
            "has_pp_handler": getattr(self, "pp_handler", None) is not None,
            "layered_enabled": bool(getattr(self, "_layered_prefill_enabled", False)),
            "single_forward": bool(getattr(self, "_single_forward", False)),
            "async_compute_enabled": self._layered_async_compute_enabled(),
            "single_forward_steps": int(counters.get("single_forward_steps") or 0),
            "pp_stage_mixed_steps": int(counters.get("pp_stage_mixed_steps") or 0),
            "compiled_group_variants": len(getattr(group_executor, "compiled_groups", {})),
            "decode_group_graph_entries": len(getattr(decode_graph_cache, "entries", {})),
            "decode_group_graph_views": len(getattr(decode_graph_cache, "views", {})),
            "layered_ready": bool(getattr(self, "_layered_prefill_v2_ready", False)),
            "layered_adapter": (getattr(self, "layered_prefill_model_adapter", None) is not None),
            "layered_buffers": (getattr(self, "_layered_input_buffers", None) is not None),
            "execute_steps": int(counters["execute_steps"]),
            "transport_frontier_steps": int(counters.get("transport_frontier_steps") or 0),
            "skip_relay_stages": bool(getattr(self, "_skip_relay_stages", False)),
            "relay_empty_steps": int(counters.get("relay_empty_steps") or 0),
            "relay_forward_steps": int(counters.get("relay_forward_steps") or 0),
            "groups": list(counters["groups"]),
            "pp_slots": list(counters["pp_slots"]),
            "activation_sources": list(counters.get("activation_sources") or []),
        }

    def _apply_scheduler_lifecycle(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Run the one-shot request lifecycle that opens execute_model."""
        self.update_pp_decode_requests()
        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        self.block_tables.apply_staged_writes()

    def _sync_ascend_num_computed_tokens_cpu(self, req_ids: list[str]) -> None:
        """Refresh Ascend's CPU num_computed mirror after lifecycle.

        ``update_requests`` only writes ``num_computed_tokens_np``.  Ascend's
        ``_update_seq_lens_cpu`` copies np→cpu for cached rows; when those rows
        are stripped for the D/P sub-batches, a stale cpu value makes
        ``seq_lens = cpu + scheduled`` one step behind and decode collapses.
        """
        for req_id in req_ids:
            req_index = self.req_states.req_id_to_index[req_id]
            self.req_states.num_computed_tokens_cpu[req_index] = int(self.req_states.num_computed_tokens_np[req_index])

    def _sync_gpu_num_computed_tokens(self, req_ids: list[str]) -> None:
        """Copy the scheduler cursor onto the GPU seq_len source.

        ``update_requests`` writes ``num_computed_tokens_np`` only. Non-sampling
        token chunks skip ``postprocess``, so the GPU cursor stays at the first
        chunk. The last chunk then looks like ``seq_len < prefill_len`` and the
        sampler drops the first decode token.
        """
        staged = False
        for req_id in req_ids:
            req_index = self.req_states.req_id_to_index.get(req_id)
            if req_index is None:
                continue
            self.req_states.num_computed_tokens.stage_write_elem(
                req_index, int(self.req_states.num_computed_tokens_np[req_index])
            )
            staged = True
        if staged:
            self.req_states.num_computed_tokens.apply_write()

    def _layered_relay_stage_action(self, layered_plan) -> str | None:
        """How this stage should relay the active layer group, if at all.

        Returns ``"empty"`` for a stage ahead of the group owner,
        ``"forward"`` for one behind it, and None when the stage has real
        work.  A relay stage computes no layer (the adapter's local range is
        empty) and writes no KV, because the layers it owns belong to groups
        that already ran.

        Real work means owning the active group, or owning the *next* one:
        ``store_req_frontiers`` tags its result ``group_id + 1``, so the
        frontier has to be left on ``owner_of(group_id + 1)``.  Everyone else
        keeps nothing anyone will read -- a stage ahead of the owner stays
        ahead for the rest of the chunk, since the owner index never
        decreases.

        Four carve-outs:

        * The sampling step.  There the last rank broadcasts sampled tokens
          and every other rank has to be inside ``sample_tokens`` to receive
          them, or its request state diverges.  On every other step
          ``LayeredPPHandlerCapture.flush`` drops a P-only payload without
          broadcasting, so there is nothing to miss.
        * The final layer group, even when it does not sample (the last group
          of a non-final token chunk).  That step clears the request's stored
          frontier on every rank, and a rank that skipped the clear would
          still be holding one when the next chunk opens at group 0 -- which
          ``_layered_prefill_activation_source`` rejects outright.
        * ``dp_size > 1``.  ``dispatch_cg_and_sync_dp`` is a collective and
          must not be desynchronised.  Layered already rejects DP > 1; this
          keeps the reason local to the skip.
        * A KV connector.  Without one the executor collects a
          ``ModelRunnerOutput`` from the last PP stage only
          (``_get_output_rank``).  With one it aggregates every rank, and the
          skipped ``kv_connector.post_forward`` would be missed.  Layered
          already rejects connectors; same reasoning, kept local.
        """
        # Boundary steps retain the completion fence and the common lifecycle.
        if getattr(layered_plan, "boundary_mode", None):
            return None
        if not self._skip_relay_stages:
            return None
        if self.parallel_config.pipeline_parallel_size <= 1:
            return None
        if self.dp_size > 1:
            return None
        if self.vllm_config.kv_transfer_config is not None:
            return None
        if getattr(layered_plan, "is_sampling_step", layered_plan.is_final_group):
            return None
        if layered_plan.is_final_group:
            return None
        rank = get_pp_group().rank_in_group
        owner = self._layered_pp_group_owner(layered_plan)
        if rank == owner:
            return None
        if rank == self._layered_next_group_owner(layered_plan):
            return None
        return "empty" if rank < owner else "forward"

    def _layered_next_group_owner(self, layered_plan) -> int | None:
        """PP rank owning the group after this one, or None if it is last.

        ``store_req_frontiers`` stamps its result with ``group_id + 1``, so
        that rank is the one stage besides the current owner which has to run
        the step for its stored frontier.
        """
        num_groups = int(layered_plan.num_groups)
        next_group = int(getattr(layered_plan, "resume_group_id", layered_plan.group_id + 1))
        if next_group >= num_groups:
            return None
        num_layers = self._num_hidden_layers()
        pp_size = self.parallel_config.pipeline_parallel_size
        key = (num_layers, num_groups, pp_size)
        cached = getattr(self, "_layered_pp_group_ranges_cache", None)
        if cached is None:
            cached = self._layered_pp_group_ranges_cache = {}
        if key not in cached:
            cached[key] = make_pp_aligned_layer_group_ranges(*key)
        layer_range = cached[key][next_group]
        for rank, (start, end) in enumerate(self._layered_pp_partitions(num_layers, pp_size)):
            if start <= layer_range.start and layer_range.end <= end:
                return rank
        return None

    def _execute_layered_relay_step(
        self,
        scheduler_output: SchedulerOutput,
        action: str,
        received,
    ):
        """Hand the activation on without touching the layer group.

        ``sample_tokens`` still owes ``scheduler.update_from_output`` a row
        per scheduled request even though nothing sampled, so stash that
        output rather than leaving the state empty.
        """
        counters = self._layered_counters()
        key = f"relay_{action}_steps"
        counters[key] = int(counters.get(key) or 0) + 1
        self.execute_model_state = None
        req_ids = list(scheduler_output.num_scheduled_tokens)
        self._layered_relay_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[[] for _ in req_ids],
        )
        if self.is_last_pp_rank:
            return None
        if action == "empty":
            return self._layered_pre_owner_placeholder(with_row_metadata=True)
        if received is None:
            raise RuntimeError("Layered relay stage has no activation to pass on")
        # Accessing ``.tensors`` on an AsyncIntermediateTensors waits for the
        # recv, so the worker's isend cannot race the incoming transfer.
        return received

    def _layered_pre_owner_placeholder(self, *, with_row_metadata: bool = False) -> IntermediateTensors:
        """Zero-row payload standing in for a skipped layer group.

        No stage reads a pre-owner activation. Empty tensors skip the transfer
        while exchanging metadata. The D/P unpacker requires zero row counts
        before the owner restores its locally stored frontier.
        """
        adapter = self.layered_prefill_model_adapter
        assert adapter is not None
        hidden, residual = adapter.make_transport_frontier(0, self.model_config.dtype, self.device)
        packed = adapter.to_intermediate_tensors(LayeredForwardOutput(hidden, residual, False))
        if with_row_metadata:
            packed.tensors["layered_pp_d_rows"] = 0
            packed.tensors["layered_pp_p_rows"] = 0
        return packed

    def _layered_nonfused_pp_payload(
        self,
        layered_plan,
        p_intermediate: IntermediateTensors | None,
    ) -> IntermediateTensors | None:
        """Trim unused P rows, retaining a one-row fence for stage-internal work."""
        if p_intermediate is None:
            return None
        owner = self._layered_pp_group_owner(layered_plan)
        compact_internal = can_compact_internal_prefill_payload(self, layered_plan)
        if not compact_internal and (owner < 0 or get_pp_group().rank_in_group >= owner):
            return p_intermediate
        # Unlike a pre-owner placeholder, PP0 has just executed an internal group.
        # Keep a non-empty send/recv to order its work before PP1 completes
        # this RPC, including when there are no Decode rows on the wire.
        rows = 1 if compact_internal else 0
        trimmed = IntermediateTensors({key: value[:rows] for key, value in p_intermediate.tensors.items()})
        trimmed.kv_connector_output = getattr(p_intermediate, "kv_connector_output", None)
        return trimmed

    def _layered_pp_partitions(self, num_layers: int, pp_size: int):
        key = (num_layers, pp_size)
        cached = getattr(self, "_layered_pp_partitions_cache", None)
        if cached is None:
            cached = self._layered_pp_partitions_cache = {}
        if key not in cached:
            cached[key] = tuple(get_pp_indices(num_layers, rank, pp_size) for rank in range(pp_size))
        return cached[key]

    def _layered_pp_group_owner(self, plan) -> int:
        pp = get_pp_group()
        configs = (
            getattr(self.model_config, "hf_text_config", None),
            getattr(self.model_config, "hf_config", None),
            self.model_config,
        )
        num_layers = int(plan.group_end)
        for config in configs:
            raw_num_layers = getattr(config, "num_hidden_layers", None)
            if isinstance(raw_num_layers, int):
                num_layers = int(raw_num_layers)
                break
        if getattr(plan, "boundary_mode", None):
            # A boundary interval can traverse multiple stages in one step.
            owners = [
                rank
                for rank, (start, end) in enumerate(self._layered_pp_partitions(num_layers, pp.world_size))
                if max(start, plan.group_start) < min(end, plan.group_end)
            ]
            if not owners:
                raise RuntimeError("Boundary interval has no local PP layers")
            return owners[0] if len(owners) == 1 else -1
        if int(plan.group_start) == 0 and int(plan.group_end) >= num_layers:
            # Full-model range: every rank runs its own stage inside this step.
            return -1
        for rank, (start, end) in enumerate(self._layered_pp_partitions(num_layers, pp.world_size)):
            if start <= plan.group_start and plan.group_end <= end:
                return rank
        raise RuntimeError(
            f"Layered group [{plan.group_start}, {plan.group_end}) is not aligned with the PP layer partition"
        )

    @staticmethod
    def _layered_pp_row_count(value, key: str) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            row_count = value
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            row_count = int(value.item())
        else:
            raise RuntimeError(f"Layered PP metadata {key} must be an integer or scalar tensor")
        if row_count < 0:
            raise RuntimeError(f"Layered PP metadata {key} cannot be negative")
        return row_count

    @staticmethod
    def _clone_intermediate_tensors(
        result: IntermediateTensors,
    ) -> IntermediateTensors:
        cloned = IntermediateTensors(
            {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in result.tensors.items()}
        )
        cloned.kv_connector_output = getattr(result, "kv_connector_output", None)
        return cloned

    @staticmethod
    def _split_layered_pp_intermediate(
        intermediate_tensors: IntermediateTensors | None,
    ) -> tuple[IntermediateTensors | None, IntermediateTensors | None]:
        if intermediate_tensors is None:
            return None, None
        tensors = intermediate_tensors.tensors
        if "layered_pp_d_rows" not in tensors or "layered_pp_p_rows" not in tensors:
            raise RuntimeError("Layered PP intermediate tensors are missing D/P row metadata")
        d_rows = NPUModelRunner._layered_pp_row_count(tensors["layered_pp_d_rows"], "layered_pp_d_rows")
        p_rows = NPUModelRunner._layered_pp_row_count(tensors["layered_pp_p_rows"], "layered_pp_p_rows")
        tensor_items = [
            (key, value)
            for key, value in tensors.items()
            if key not in ("layered_pp_d_rows", "layered_pp_p_rows") and isinstance(value, torch.Tensor)
        ]
        if not tensor_items:
            raise RuntimeError("Layered PP intermediate tensors contain no tensors")
        for key, value in tensor_items:
            if value.ndim == 0 or d_rows + p_rows > value.shape[0]:
                raise RuntimeError(
                    f"Layered PP metadata exceeds {key} row dimension: "
                    f"{d_rows}+{p_rows}>{value.shape[0] if value.ndim else 0}"
                )
        d_tensors = {key: value[:d_rows] for key, value in tensor_items}
        p_tensors = {key: value[d_rows : d_rows + p_rows] for key, value in tensor_items}
        return IntermediateTensors(d_tensors), IntermediateTensors(p_tensors)

    @staticmethod
    def _combine_layered_pp_intermediate(
        d_intermediate: IntermediateTensors | None,
        p_intermediate: IntermediateTensors | None,
    ) -> IntermediateTensors:
        if d_intermediate is None and p_intermediate is None:
            raise RuntimeError("Layered PP pack requires D or P intermediates")
        d_tensors = d_intermediate.tensors if d_intermediate is not None else {}
        p_tensors = p_intermediate.tensors if p_intermediate is not None else {}
        keys = dict.fromkeys((*d_tensors, *p_tensors))
        tensors: dict = {}
        for key in keys:
            d_value = d_tensors.get(key)
            p_value = p_tensors.get(key)
            if isinstance(d_value, torch.Tensor) and isinstance(p_value, torch.Tensor):
                tensors[key] = torch.cat((d_value, p_value), dim=0)
            elif isinstance(d_value, torch.Tensor):
                tensors[key] = d_value
            elif isinstance(p_value, torch.Tensor):
                tensors[key] = p_value
        d_rows = (
            0
            if d_intermediate is None
            else next(
                (value.shape[0] for value in d_tensors.values() if isinstance(value, torch.Tensor)),
                0,
            )
        )
        p_rows = (
            0
            if p_intermediate is None
            else next(
                (value.shape[0] for value in p_tensors.values() if isinstance(value, torch.Tensor)),
                0,
            )
        )
        if not tensors:
            raise RuntimeError("Layered PP intermediate tensors contain no tensors")
        # Control fields ride the existing object metadata message; only
        # activations need tensor P2P. LP unpacks these before model slicing.
        tensors["layered_pp_d_rows"] = int(d_rows)
        tensors["layered_pp_p_rows"] = int(p_rows)
        return IntermediateTensors(tensors)

    def _prepare_layered_p_activation(
        self,
        layered_plan,
        *,
        req_ids: list[str] | None = None,
        num_tokens_padded: int,
        inputs_embeds,
        intermediate_tensors: IntermediateTensors | None,
    ):
        """Pick frontier / embed / PP transport inputs for the P (or mixed) batch."""
        layered_adapter = self.layered_prefill_model_adapter
        assert layered_adapter is not None
        if req_ids is None:
            req_ids = list(layered_plan.prefill_req_ids)
        if not req_ids:
            raise RuntimeError("Layered Prefill activation has no request ids")
        if getattr(layered_plan, "boundary_mode", None):
            return self._prepare_boundary_p_activation(
                layered_plan, req_ids, num_tokens_padded, inputs_embeds, intermediate_tensors
            )
        pp = get_pp_group()
        owner = self._layered_pp_group_owner(layered_plan)
        if owner < 0:
            # The plan covers every stage. Rank 0 embeds; later ranks consume
            # the activation the previous stage just sent.
            if pp.rank_in_group == 0 and layered_plan.group_id == 0:
                return req_ids, None, inputs_embeds, None, "embed"
            if intermediate_tensors is None:
                raise RuntimeError("Layered PP stage did not receive activation for a full-model layer range")
            return req_ids, None, None, intermediate_tensors, "pp_recv"
        owner_has_frontier = layered_plan.group_id > 0 and pp.rank_in_group == owner
        if owner_has_frontier:
            if len(req_ids) > 1:
                frontier_tuple = concat_req_frontiers(
                    self.layered_prefill_state,
                    req_ids,
                    expected_group_id=layered_plan.group_id,
                    num_tokens_padded=num_tokens_padded,
                )
                return req_ids, frontier_tuple, None, None, "frontier_mixed"
            req_id = req_ids[0]
            frontier = self.layered_prefill_state.get(req_id)
            if frontier is None:
                raise RuntimeError(f"Missing layered activation frontier for request {req_id}")
            if frontier.group_id != layered_plan.group_id:
                raise RuntimeError(
                    f"Layered frontier group mismatch for {req_id}: expected "
                    f"{layered_plan.group_id}, got {frontier.group_id}"
                )
            return (
                req_ids,
                pad_activation_rows(
                    frontier.hidden_states,
                    frontier.residual,
                    num_tokens_padded,
                ),
                None,
                None,
                "frontier",
            )

        req_id = req_ids[0]
        frontier = self.layered_prefill_state.get(req_id)
        if frontier is not None and pp.rank_in_group == owner:
            raise RuntimeError(f"Unexpected layered frontier for group 0 request {req_id}")
        frontier_tuple = None
        if layered_plan.group_id == 0 and pp.rank_in_group == 0:
            initial_inputs_embeds = inputs_embeds
            source = "embed"
        elif pp.rank_in_group < owner and layered_plan.group_id > 0:
            # D5: ranks before the group owner still participate in the
            # PP send/recv chain.  A dummy activation keeps NCCL matched
            # without consuming a leftover local frontier.
            frontier_tuple = layered_adapter.make_transport_frontier(
                num_tokens_padded,
                self.model_config.dtype,
                self.device,
            )
            initial_inputs_embeds = None
            source = "transport_frontier"
        else:
            initial_inputs_embeds = None
            source = "pp_recv" if pp.rank_in_group > owner else "none"
        receives_pp_activation = pp.rank_in_group > owner
        if receives_pp_activation and intermediate_tensors is None:
            raise RuntimeError("Layered PP stage did not receive P activation")
        return (
            req_ids,
            frontier_tuple,
            initial_inputs_embeds,
            intermediate_tensors if receives_pp_activation else None,
            source,
        )

    def _prepare_boundary_p_activation(self, plan, req_ids, padded, embeds, received):
        """Restore only the interval's entry; later stages consume this step's wire."""
        adapter = self.layered_prefill_model_adapter
        start = int(plan.group_start)
        if start >= adapter.end_layer:
            return (
                req_ids,
                adapter.make_transport_frontier(padded, self.model_config.dtype, self.device),
                None,
                None,
                "transport_frontier",
            )
        if start < adapter.start_layer:
            if received is None:
                raise RuntimeError("Boundary stage requires this step's PP activation")
            return req_ids, None, None, received, "pp_recv"
        if start == 0:
            if any(self.layered_prefill_state.get(req_id) is not None for req_id in req_ids):
                raise RuntimeError("Unexpected boundary frontier at chunk entry")
            return req_ids, None, embeds, None, "embed"
        for req_id in req_ids:
            frontier = self.layered_prefill_state.get(req_id)
            if (
                frontier is None
                or frontier.group_id != plan.group_id
                or frontier.query_len != plan.query_tokens[req_id]
                or getattr(frontier, "cohort_id", None) != plan.cohort_id
            ):
                raise RuntimeError(f"Missing or mismatched boundary frontier for {req_id}")
        if len(req_ids) == 1:
            saved = self.layered_prefill_state.get(req_ids[0])
            frontier = pad_activation_rows(saved.hidden_states, saved.residual, padded)
        else:
            frontier = concat_req_frontiers(
                self.layered_prefill_state,
                req_ids,
                expected_group_id=plan.group_id,
                num_tokens_padded=padded,
            )
        return req_ids, frontier, None, None, "frontier"

    @staticmethod
    def _layered_prefill_moe_layer_offset(all_moe_layers: list[str], layer_start: int) -> int:
        """Return the MoE custom-op layer offset for a Transformer layer boundary."""
        return sum(extract_layer_index(layer_name) < layer_start for layer_name in all_moe_layers)

    def _set_layered_prefill_moe_layer_offset(self, layer_start: int) -> None:
        forward_context = get_forward_context()
        all_moe_layers = forward_context.all_moe_layers
        # Empty list means the compile inventory was never filled; treat as unset
        # so MoE ops fall back to baking layer names rather than indexing [].
        if not all_moe_layers:
            return
        offset = self._layered_prefill_moe_layer_offset(all_moe_layers, layer_start)
        forward_context.moe_layer_index = offset

    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        """Pad query tokens to a TP multiple when DSA-CP or native SP is on.

        Same contract as V1 ``NPUModelRunner._pad_for_sequence_parallelism``:
        scheduler bookkeeping stays on the logical length.
        """
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if enable_sp(self.vllm_config) or enable_dsa_cp():
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _prepare_layered_batch(self, scheduler_output, *, build_attention_metadata=True):
        if scheduler_output.total_num_scheduled_tokens == 0:
            raise RuntimeError("Layered Prefill P sub-batch has zero tokens")

        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len = max(scheduler_output.num_scheduled_tokens.values())
        batch_req_state, uniform_tok_count = self.gather_batch_req_state(scheduler_output, dummy_run=False)
        assert batch_req_state is not None
        num_toks = batch_req_state.num_tokens
        max_query_len = int(batch_req_state.num_scheduled_tokens.max())

        # Prefill layer groups stay eager (migration plan §2 / V4).
        batch_desc, dp_sync = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_toks,
            uniform_tok_count,
            self.dp_size,
            self.dp_rank,
            need_eager=True,
            num_active_loras=0,
        )
        num_tokens_across_dp = dp_sync.num_tokens_across_dp if dp_sync is not None else None
        if batch_desc.cg_mode != CUDAGraphMode.NONE:
            raise RuntimeError(f"Layered Prefill P sub-batch must stay eager (got cudagraph mode {batch_desc.cg_mode})")
        if batch_desc.num_tokens == 0:
            raise RuntimeError("Layered Prefill P sub-batch dispatched zero tokens")
        # Match V1 / origin dsa_cp: pad the physical eager batch to a TP
        # multiple while scheduler_output keeps the logical prompt length.
        # KV/attention use the logical count; MoE collectives use the padded
        # width. Pad rows get slot_id=-1 (compute_slot_mappings) and zeroed
        # positions.
        padded_toks = self._pad_for_sequence_parallelism(num_toks)
        if padded_toks != batch_desc.num_tokens:
            batch_desc = replace(batch_desc, num_tokens=padded_toks)

        input_batch = self.prepare_inputs(scheduler_output, batch_req_state, batch_desc)
        if padded_toks != input_batch.num_tokens_after_padding:
            raise RuntimeError(
                "Layered Prefill P pad width mismatch: batch_desc="
                f"{padded_toks}, input_batch={input_batch.num_tokens_after_padding}"
            )
        if input_batch.num_tokens_after_padding > input_batch.num_tokens:
            pad_from = input_batch.num_tokens
            pad_to = input_batch.num_tokens_after_padding
            self.input_buffers.positions[pad_from:pad_to].fill_(0)
        if not build_attention_metadata:
            # Sampling on a non-owner rank consumes the ordinary InputBatch,
            # but runs no attention and must not replace D backend metadata.
            return SimpleNamespace(
                input_batch=input_batch,
                attn_metadata=None,
                slot_mappings_by_layer=None,
                block_tables=None,
                slot_mappings=None,
                batch_desc=batch_desc,
                num_tokens_across_dp=num_tokens_across_dp,
                dp_sync=dp_sync,
            )
        block_tables, slot_mappings = self.prepare_attn(input_batch)
        self.model_state.preprocess_state(
            input_batch,
            block_tables,
            self.kv_cache_config,
            self.req_states.num_computed_tokens.gpu,
        )

        slot_mappings_by_layer = build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)
        attn_metadata = self.model_state.prepare_attn(
            input_batch,
            batch_desc.cg_mode,
            block_tables,
            slot_mappings,
            self.attn_groups,
            self.kv_cache_config,
            for_capture=False,
            num_input_tokens=padded_toks if padded_toks > num_toks else None,
        )

        return SimpleNamespace(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            batch_desc=batch_desc,
            num_tokens_across_dp=num_tokens_across_dp,
            dp_sync=dp_sync,
        )

    def _run_layered_prefill_subbatch(
        self,
        scheduler_output: SchedulerOutput,
        layered_plan,
        intermediate_tensors: IntermediateTensors | None = None,
        *,
        moe_comm_token_count: int | None = None,
    ) -> tuple[ExecuteModelState, LayeredForwardOutput]:
        """Eager layer-range forward: prepare_inputs → prepare_attn → adapter."""
        # Honor the transport fence even if local capability disables the
        # ordinary non-owner fastpath.
        fenced = run_internal_prefill_fence(self, scheduler_output, layered_plan, intermediate_tensors)
        if fenced is not None:
            return fenced
        if getattr(self, "_layered_nonowner_fastpath", False):
            nonowner = run_nonowner_prefill(self, scheduler_output, layered_plan, intermediate_tensors)
            if nonowner is not None:
                return nonowner
        prepared = self._prepare_layered_batch(scheduler_output)
        input_batch = prepared.input_batch
        attn_metadata = prepared.attn_metadata
        slot_mappings_by_layer = prepared.slot_mappings_by_layer
        batch_desc = prepared.batch_desc
        num_tokens_across_dp = prepared.num_tokens_across_dp
        num_toks = scheduler_output.total_num_scheduled_tokens

        input_ids = input_batch.input_ids
        inputs_embeds = None
        positions = input_batch.positions
        num_tokens_padded = input_batch.num_tokens_after_padding

        batch_descriptor = BatchDescriptor(
            num_tokens=num_tokens_padded,
            has_lora=False,
            num_active_loras=0,
        )
        layered_adapter = self.layered_prefill_model_adapter
        assert layered_adapter is not None
        num_reqs = input_batch.num_reqs
        batch_req_ids = list(input_batch.req_ids[:num_reqs])
        run_start = int(layered_plan.group_start)
        run_end = int(layered_plan.group_end)
        run_final = bool(layered_plan.is_final_group)
        (
            _act_req_ids,
            frontier_tuple,
            initial_inputs_embeds,
            p_pp_intermediate,
            activation_source,
        ) = self._prepare_layered_p_activation(
            layered_plan,
            req_ids=batch_req_ids,
            num_tokens_padded=num_tokens_padded,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        self._record_layered_activation(layered_plan, activation_source)
        use_group_executor = hasattr(self, "_layered_group_executor") and LayeredV2SingleForward.supported(
            self, allow_pipeline_parallel=True
        )
        local_start = max(run_start, layered_adapter.start_layer)
        local_end = min(run_end, layered_adapter.end_layer)
        with set_ascend_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_toks,
            aclgraph_runtime_mode=batch_desc.cg_mode,
            batch_descriptor=batch_descriptor,
            model_instance=self.model,
            skip_compiled=not use_group_executor,
            moe_comm_token_count=num_toks if moe_comm_token_count is None else moe_comm_token_count,
        ):
            ctx = get_forward_context()
            ctx.slot_mapping = slot_mappings_by_layer
            ctx.is_padding = input_batch.is_padding
            self._set_layered_prefill_moe_layer_offset(local_start)
            self.kv_connector.pre_forward(scheduler_output)

            if use_group_executor:
                layered_adapter._validate_range(run_start, run_end)
                activation = layered_adapter._prepare_initial_state(
                    input_ids=input_ids,
                    frontier=frontier_tuple,
                    inputs_embeds=initial_inputs_embeds,
                    intermediate_tensors=p_pp_intermediate,
                )
                if local_start < local_end:
                    activation = self._layered_group_executor.forward_mixed(
                        local_start,
                        local_end,
                        activation,
                        input_ids,
                        positions[:num_tokens_padded],
                    )
                if run_final and layered_adapter.end_layer == layered_adapter.num_hidden_layers:
                    activation = layered_adapter._finalize(*activation)
                layered_output = LayeredForwardOutput(
                    hidden_states=activation[0],
                    residual=activation[1],
                    is_final_layer=run_final,
                )
            else:
                layered_output = layered_adapter.forward(
                    input_ids=input_ids,
                    positions=positions[:num_tokens_padded],
                    layer_start=run_start,
                    layer_end=run_end,
                    frontier=frontier_tuple,
                    inputs_embeds=initial_inputs_embeds,
                    intermediate_tensors=p_pp_intermediate,
                )

        # mem_after = int(torch.npu.memory_allocated(self.device))
        # peak_after = int(
        #     torch.npu.memory_stats(self.device).get("allocated_bytes.all.peak", 0)
        # )

        actual_rows = int(layered_output.hidden_states.shape[0])
        if actual_rows != num_tokens_padded:
            raise RuntimeError(
                "Layered model returned a hidden-state row count that "
                "does not match the P query batch: "
                f"actual_rows={actual_rows} expected_rows={num_tokens_padded} "
                f"actual_hidden={tuple(layered_output.hidden_states.shape)} "
                f"actual_residual="
                f"{None if layered_output.residual is None else tuple(layered_output.residual.shape)} "
                f"group={layered_plan.group_id}/{layered_plan.num_groups} "
                f"layers=[{run_start},{run_end}) logical_query={num_toks} "
                f"reqs={batch_req_ids} source={activation_source}"
            )
        if layered_output.is_final_layer != run_final:
            raise RuntimeError("Layered model final-layer status does not match the plan")

        if run_final:
            for req_id in batch_req_ids:
                self.layered_prefill_state.clear(req_id)
            hidden_states = layered_output.hidden_states
        else:
            store_ids = batch_req_ids
            if getattr(layered_plan, "boundary_mode", None):
                # PP0's stage exit is not the global interval's frontier.
                # Retain state only at the rank owning the next real layer.
                if get_pp_group().rank_in_group != self._layered_next_group_owner(layered_plan):
                    self.layered_prefill_state.clear_many(batch_req_ids)
                    store_ids = []
            if store_ids:
                store_req_frontiers(
                    self.layered_prefill_state,
                    store_ids,
                    np.array(input_batch.query_start_loc_np, copy=True),
                    layered_output.hidden_states,
                    layered_output.residual,
                    getattr(layered_plan, "resume_group_id", layered_plan.group_id + 1),
                    batch_req_ids=batch_req_ids,
                    keep_ids=store_ids,
                    reuse_existing=use_group_executor,
                    cohort_id=layered_plan.cohort_id if getattr(layered_plan, "boundary_mode", None) else None,
                )
            hidden_states = None

        return (
            ExecuteModelState(
                input_batch=input_batch,
                attn_metadata=attn_metadata,
                slot_mappings_by_layer=slot_mappings_by_layer,
                hidden_states=hidden_states,
                aux_hidden_states=None,
                finished_req_ids=scheduler_output.finished_req_ids,
                dp_sync=prepared.dp_sync,
                ec_connector_output=None,
                routed_experts=None,
                cudagraph_stats=None,
            ),
            layered_output,
        )

    def _sample_layered_v2(self, grammar_output, state: LayeredV2ExecuteModelState):
        outputs: list[ModelRunnerOutput] = []
        p_req_ids = list(state.p_state.input_batch.req_ids) if state.p_state is not None else []
        with self._layered_pp_handler_capture() as capture:
            try:
                if state.d_state is not None:
                    self.execute_model_state = state.d_state
                    active_grammar = grammar_output
                    if grammar_output is not None and not any(
                        req_id in grammar_output.structured_output_request_ids
                        for req_id in state.d_state.input_batch.req_ids
                    ):
                        active_grammar = None
                    output = self._sample_tokens_normal(active_grammar)
                    if isinstance(output, AsyncOutput):
                        output = output.get_output()
                    if output is None:
                        output = EMPTY_MODEL_RUNNER_OUTPUT
                    if not isinstance(output, ModelRunnerOutput):
                        raise RuntimeError("Layered Prefill D sampling returned PP tensors")
                    outputs.append(output)

                if state.p_state is not None:
                    if state.sample_p:
                        # Last rank samples; non-last receives.  Both go through
                        # sample_tokens so token progress and the deferred
                        # PPHandler slot stay in lockstep.  Capture merges D ∪ P
                        # into one broadcast/receive (V6).
                        # On a non-final prompt chunk the prefill rows have no
                        # logits yet; keep the decode tokens and drop those rows.
                        # The final chunk does sample, including any Decode
                        # to reach this call.
                        self.execute_model_state = state.p_state
                        active_grammar = grammar_output
                        if grammar_output is not None and not any(
                            req_id in grammar_output.structured_output_request_ids
                            for req_id in state.p_state.input_batch.req_ids
                        ):
                            active_grammar = None
                        output = self._sample_tokens_normal(active_grammar)
                        if isinstance(output, AsyncOutput):
                            output = output.get_output()
                        if output is None:
                            output = EMPTY_MODEL_RUNNER_OUTPUT
                        if not isinstance(output, ModelRunnerOutput):
                            raise RuntimeError("Layered Prefill P sampling returned PP tensors")
                        outputs.append(output)
                    else:
                        # Non-sampling step (intermediate layer group, or the
                        # last group of a non-final prompt chunk): skip sampler,
                        # token-progress append, and the PPHandler slot.  Still
                        # emit req_ids so scheduler.update_from_output can find
                        # every scheduled request (empty sampled tokens).
                        finished = state.p_state.finished_req_ids
                        self.execute_model_state = None
                        kv_out = (
                            None
                            if state.single_forward and state.d_state is not None
                            else self.kv_connector.post_forward(finished)
                        )
                        outputs.append(
                            ModelRunnerOutput(
                                req_ids=p_req_ids,
                                req_id_to_index={req_id: i for i, req_id in enumerate(p_req_ids)},
                                sampled_token_ids=[[] for _ in p_req_ids],
                                kv_connector_output=kv_out,
                            )
                        )
            finally:
                self.execute_model_state = None
            if capture is not None:
                slot = capture.flush(
                    sample_p=state.sample_p,
                    p_req_ids=set(p_req_ids),
                    plan=getattr(state.scheduler_output, "ring_step_plan", None),
                )
                self._record_layered_pp_slot(slot)

        if not outputs:
            output = EMPTY_MODEL_RUNNER_OUTPUT
        elif len(outputs) == 1:
            output = outputs[0]
        else:
            output = self._merge_layered_v2_outputs(state.scheduler_output, outputs)
        return self._complete_layered_output(output, state)

    @staticmethod
    def _merge_layered_v2_outputs(
        scheduler_output: SchedulerOutput,
        outputs: list[ModelRunnerOutput],
    ) -> ModelRunnerOutput:
        by_req_id: dict[str, tuple[ModelRunnerOutput, int]] = {}
        for output in outputs:
            for index, req_id in enumerate(output.req_ids):
                by_req_id[req_id] = (output, index)
        req_ids = list(scheduler_output.num_scheduled_tokens)
        sampled_token_ids = [
            by_req_id[req_id][0].sampled_token_ids[by_req_id[req_id][1]] if req_id in by_req_id else []
            for req_id in req_ids
        ]
        prompt_logprobs: dict = {}
        num_nans: dict = {}
        for output in outputs:
            prompt_logprobs.update(output.prompt_logprobs_dict)
            if output.num_nans_in_logits:
                num_nans.update(output.num_nans_in_logits)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=next((o.logprobs for o in outputs if o.logprobs is not None), None),
            prompt_logprobs_dict=prompt_logprobs,
            pooler_output=[],
            kv_connector_output=next(
                (o.kv_connector_output for o in outputs if o.kv_connector_output),
                None,
            ),
            ec_connector_output=next(
                (o.ec_connector_output for o in outputs if o.ec_connector_output),
                None,
            ),
            num_nans_in_logits=num_nans or None,
            cudagraph_stats=next(
                (o.cudagraph_stats for o in outputs if o.cudagraph_stats),
                None,
            ),
            routed_experts=None,
        )

    def _profile_layered_frontier_budget(self) -> None:
        """Reserve cross-group frontiers for every in-flight PP cohort.

        Startup ``profile_run`` uses dummy forward and never enters the
        layered store/restore path, so without this allocation the KV pool
        is sized as if the frontier did not exist. Shape matches the live
        DeepSeek-V4 ``[T, hc_mult, H]`` frontier (no residual) or the
        standard ``[T, H]`` + residual pair.
        """
        self.layered_frontier_profile_bytes = 0
        if not self._layered_prefill_enabled:
            return
        from vllm_ascend.models.layered_prefill import (
            DeepseekV4LayeredPrefillAdapter,
        )

        adapter = self.layered_prefill_model_adapter
        hidden_size = int(self.model_config.get_hidden_size())
        hc_mult = 1
        reserve_residual = True
        if isinstance(adapter, DeepseekV4LayeredPrefillAdapter):
            hc_mult = int(adapter.backbone.hc_mult)
            reserve_residual = False
        elif adapter is not None:
            backbone = getattr(adapter, "backbone", None)
            if backbone is not None and hasattr(backbone, "hc_mult"):
                hc_mult = int(backbone.hc_mult)
        cohort_slots = max(1, int(self.parallel_config.pipeline_parallel_size))
        frontier_rows = int(self.max_num_tokens) * cohort_slots
        if hc_mult > 1:
            shape: tuple[int, ...] = (
                frontier_rows,
                hc_mult,
                hidden_size,
            )
        else:
            shape = (frontier_rows, hidden_size)
        peak_before = int(torch.npu.memory_stats(self.device).get("allocated_bytes.all.peak", 0))
        frontier = torch.empty(shape, dtype=self.dtype, device=self.device)
        residual = torch.empty(shape, dtype=self.dtype, device=self.device) if reserve_residual else None
        torch.npu.synchronize()
        bytes_reserved = frontier.numel() * frontier.element_size()
        if residual is not None:
            bytes_reserved += residual.numel() * residual.element_size()
        peak_after = int(torch.npu.memory_stats(self.device).get("allocated_bytes.all.peak", 0))
        # Profile frees the dummy forward before this allocation, so the
        # high-water mark is max(forward, frontier), not their sum. The live
        # path keeps the frontier while the next group forwards, so subtract
        # the part of ``bytes_reserved`` that did not raise the peak.
        peak_included = max(0, peak_after - peak_before)
        extra = max(0, bytes_reserved - peak_included)
        self.layered_frontier_profile_bytes = extra
        del frontier
        del residual

    def _reset_input_padding(self, num_tokens: int, padded_tokens: int) -> None:
        self.input_buffers.is_padding[:num_tokens].fill_(False)
        if padded_tokens > num_tokens:
            self.input_buffers.is_padding[num_tokens:padded_tokens].fill_(True)


@contextmanager
def graph_manager_wrapper(model_runner):
    """Context manager to override graph manager."""
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    def factory(  # type: ignore[misc]
        vllm_config: VllmConfig,
        device: torch.device,
        cudagraph_mode: CUDAGraphMode,
        decode_query_len: int,
        lora_capture_cases: list[int] | None = None,
        varlen_decode: bool = False,
        ubatch_runner: Any = None,  # vLLM main (#51700)
    ):
        return ModelAclGraphManager(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            model_runner,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            ubatch_runner=ubatch_runner,
        )

    try:
        vllm_model_runner.ModelCudaGraphManager = factory
        yield
    finally:
        vllm_model_runner.ModelCudaGraphManager = original_graph_manager
