# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Token-level Prefill burst admission for low-bandwidth PP.

The controller keeps Prefill admission open until the running request
capacity is reached, latches a Prefill HOLD while Decode drains, and releases
the next burst below the active-Decode watermark. HOLD only blocks new
admission; running Prefill chunks continue. New arrivals can join an open burst. The upstream vLLM scheduler
still owns request ordering, KV allocation, preemption, and per-request
``num_scheduled_tokens`` used by both model runner versions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.ascend_config import PrefillAdmissionConfig, init_ascend_config


@dataclass(frozen=True)
class PrefillAdmissionDecision:
    """Admission constraint to apply to one token-level scheduling step."""

    throttle_prefills: bool
    token_budget: int | None
    reason: str
    pending_prefill_ids: frozenset[str]


class PrefillAdmissionController:
    """Compute Prefill bursts with a capacity-triggered Prefill hold."""

    def __init__(
        self,
        config: PrefillAdmissionConfig,
        pipeline_parallel_size: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.decode_low_watermark = config.decode_low_watermark or pipeline_parallel_size
        self.prefill_cooldown_s = config.prefill_burst_cooldown_ms / 1000.0
        self.prefill_tokens_per_pp_bubble = config.prefill_tokens_per_pp_bubble
        # Retained for constructor and config compatibility. Admission is now
        # controlled by the capacity-triggered HOLD latch, not wall-clock time.
        self._clock = clock
        self._prefill_hold = False

    @staticmethod
    def _decode_token_demand(request: Request) -> int:
        return max(
            0,
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens,
        )

    def decide(
        self,
        running_requests: Iterable[Request],
        pending_prefills: Iterable[Request],
        *,
        all_pending_prefills: Callable[[], Iterable[Request]] | None = None,
        waiting_prefills: Iterable[Request] | None = None,
        max_prefill_slots: int | None = None,
        scheduler_step: int,
        max_token_budget: int,
    ) -> PrefillAdmissionDecision:
        """Keep admission open until capacity, without freezing arrivals."""
        running_requests = tuple(running_requests)
        pending_prefill_ids = frozenset(
            request.request_id for request in pending_prefills
        )
        active_decodes = [
            request
            for request in running_requests
            if not request.is_prefill_chunk
            and self._decode_token_demand(request) > 0
        ]
        active_decode_batch_size = len(active_decodes)
        running_full = (
            max_prefill_slots is not None
            and len(running_requests) >= max_prefill_slots
        )
        low_decode_release = active_decode_batch_size < self.decode_low_watermark
        hold_released = self._prefill_hold and low_decode_release
        if low_decode_release:
            self._prefill_hold = False
        elif running_full:
            self._prefill_hold = True

        if not pending_prefill_ids:
            return PrefillAdmissionDecision(
                self._prefill_hold, None, "no_prefill", pending_prefill_ids
            )
        if self._prefill_hold:
            return PrefillAdmissionDecision(
                True, None, "running_full_hold", pending_prefill_ids
            )

        decode_token_budget = min(
            max_token_budget,
            sum(
                self._decode_token_demand(request)
                for request in active_decodes
                if scheduler_step >= request.next_decode_eligible_step
            ),
        )
        bubble_capacity = max(
            self.decode_low_watermark - active_decode_batch_size, 1
        )
        token_budget = min(
            max_token_budget,
            decode_token_budget
            + bubble_capacity * self.prefill_tokens_per_pp_bubble,
        )
        return PrefillAdmissionDecision(
            False,
            token_budget,
            "low_decode_release" if hold_released else "open",
            pending_prefill_ids,
        )

    def observe(
        self,
        decision: PrefillAdmissionDecision,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Compatibility hook for scheduling observers."""


class _PrefillAdmissionSchedulerMixin:
    """Apply controller decisions while delegating scheduling to vLLM."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        scheduler_extension_config = init_ascend_config(self.vllm_config).scheduler_config
        admission_config = scheduler_extension_config.prefill_admission_config
        self._prefill_admission_controller: PrefillAdmissionController | None = None

        # Keep ShortRequestFirst composable with the admission constraint. The
        # installer is idempotent when a parent scheduler already installed it.
        short_request_config = scheduler_extension_config.short_request_first_config
        if short_request_config.enabled:
            from vllm_ascend.core.short_request_first_scheduler import install_short_request_first_waiting_queue

            install_short_request_first_waiting_queue(
                self,
                threshold=short_request_config.threshold,
                long_max_wait_ms=short_request_config.long_max_wait_ms,
            )

        if not admission_config.enabled:
            return

        self._prefill_admission_controller = PrefillAdmissionController(
            admission_config,
            self.parallel_config.pipeline_parallel_size,
        )
        # logger.info(
        #     "Prefill admission capacity HOLD enabled: "
        #     "decode_low_watermark=%d, prefill_tokens_per_pp_bubble=%d, "
        #     "configured_cooldown_ms=%.3f (compatibility only)",
        #     self._prefill_admission_controller.decode_low_watermark,
        #     admission_config.prefill_tokens_per_pp_bubble,
        #     admission_config.prefill_burst_cooldown_ms,
        # )

    @staticmethod
    def _waiting_request_needs_prefill(request: Request) -> bool:
        return request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED) and (
            request.num_computed_tokens < request.num_tokens - 1
        )

    def _pending_prefills(self) -> list[Request]:
        pending: dict[str, Request] = {
            request.request_id: request for request in self.running if request.is_prefill_chunk
        }
        for request_queue in (self.waiting, self.skipped_waiting):
            for request in request_queue:
                if self._waiting_request_needs_prefill(request):
                    pending.setdefault(request.request_id, request)
                    # Queue order determines admission order. Tracking the
                    # first eligible Prefill avoids an O(waiting) hot-path scan
                    # while still aging the next request that can make progress.
                    break
        return list(pending.values())

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        controller = self._prefill_admission_controller
        if controller is None:
            return super().schedule(throttle_prefills)

        max_prefill_slots = max(
            0,
            self.max_num_running_reqs
            - getattr(self, "num_waiting_for_streaming_input", 0),
        )
        decision = controller.decide(
            self.running,
            self._pending_prefills(),
            max_prefill_slots=max_prefill_slots,
            scheduler_step=self.current_step + 1,
            max_token_budget=self.max_num_scheduled_tokens,
        )
        original_token_budget = self.max_num_scheduled_tokens
        if decision.token_budget is not None:
            self.max_num_scheduled_tokens = decision.token_budget
        try:
            scheduler_output = super().schedule(
                throttle_prefills,
                admit_new_prefills=not decision.throttle_prefills,
                separate_prefill_quota=True,
            )
        finally:
            self.max_num_scheduled_tokens = original_token_budget

        controller.observe(decision, scheduler_output)
        return scheduler_output


class PrefillAdmissionScheduler(_PrefillAdmissionSchedulerMixin, Scheduler):
    """Synchronous vLLM scheduler with prefill admission throttling."""


class PrefillAdmissionAsyncScheduler(_PrefillAdmissionSchedulerMixin, AsyncScheduler):
    """Asynchronous vLLM scheduler with prefill admission throttling."""
