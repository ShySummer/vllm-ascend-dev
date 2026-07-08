#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from typing import Any

import torch
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class AscendSpecDecodeBaseProposer310(AscendSpecDecodeBaseProposer):
    """310P proposer overrides for NPU-specific spec-decode workarounds."""

    def _copy_input_ids_first_pass_310p(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        num_tokens: int,
    ) -> None:
        # 310P occasionally faults on the tiny device slice/clone/index-write
        # sequence used to shift MTP input_ids. Stage the small id vector on CPU
        # and upload it once to keep the graph input buffer deterministic.
        input_ids_cpu = target_token_ids[:num_tokens].detach().cpu()
        if num_tokens > 1:
            input_ids_cpu[:-1] = input_ids_cpu[1:].clone()
        input_ids_cpu[-1] = 0
        token_indices_cpu = token_indices_to_sample.detach().cpu().to(torch.long)
        input_ids_cpu[token_indices_cpu] = next_token_ids.detach().cpu()
        self.input_ids[:num_tokens].copy_(input_ids_cpu, non_blocking=True)

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ):
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        cu_num_draft_tokens_cpu = spec_decode_metadata.cu_num_draft_tokens.detach().cpu().to(torch.int32)
        valid_count_cpu = valid_sampled_tokens_count.detach().cpu().to(torch.int32)[:num_reqs]
        num_draft_tokens_cpu = torch.cat(
            [
                cu_num_draft_tokens_cpu[0:1],
                cu_num_draft_tokens_cpu[1:num_reqs] - cu_num_draft_tokens_cpu[: num_reqs - 1],
            ]
        )
        num_rejected_tokens_cpu = torch.where(
            num_draft_tokens_cpu > 0,
            num_draft_tokens_cpu + 1 - valid_count_cpu,
            torch.zeros_like(num_draft_tokens_cpu),
        )
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        token_indices_to_sample_cpu = (
            query_start_loc_cpu[1 : num_reqs + 1].to(torch.int32) - 1 - num_rejected_tokens_cpu
        )

        num_rejected_tokens_gpu = num_rejected_tokens_cpu.to(device, non_blocking=True)
        token_indices_to_sample = token_indices_to_sample_cpu.to(device, non_blocking=True)

        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]

        spec_common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=common_attn_metadata.seq_lens_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens if self.pcp_size > 1 else total_num_tokens,
            num_input_tokens=common_attn_metadata.num_input_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            slot_mapping_cpu=common_attn_metadata.slot_mapping_cpu,
            positions=common_attn_metadata.positions,
            positions_cpu=common_attn_metadata.positions_cpu,
            attn_state=self.runner.attn_state,
            decode_token_per_req=self.runner.decode_token_per_req,
            num_computed_tokens_cpu=common_attn_metadata.num_computed_tokens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            is_prefilling=common_attn_metadata.is_prefilling,
            max_seq_len=0,
        )
        return spec_common_attn_metadata, token_indices, token_indices_to_sample, num_rejected_tokens_gpu

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        if not self.needs_extra_input_slots:
            # 310P workaround for MTP:
            # The NPU implementation of the slice assign
            #   self.input_ids[:num_tokens-1] = target_token_ids[1:]
            # can corrupt the tail element (index num_tokens-1) of the
            # persistent drafter input_ids buffer. We save/restore it to
            # avoid feeding garbage to the draft model or later GatherV2.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]

            # Protected shift (310P specific)
            AscendSpecDecodeBaseProposer310._copy_input_ids_first_pass_310p(
                self,
                target_token_ids,
                next_token_ids,
                token_indices_to_sample,
                num_tokens,
            )

            assert self.runner is not None

            # 310P does not support PCP/DCP, so we skip all PCP handling.
            ori_token_indices_to_sample = None
            query_lens_d = None

            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]

            self._set_positions(num_tokens, target_positions)
            self.hidden_states[:num_tokens] = target_hidden_states.view(num_tokens, -1)

            return num_tokens, token_indices_to_sample, cad, (query_lens_d, ori_token_indices_to_sample)
        return super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
