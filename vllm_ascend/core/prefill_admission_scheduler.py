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
admission; running Prefill chunks continue. The upstream vLLM scheduler
still owns request ordering, KV allocation, preemption, and per-request
``num_scheduled_tokens`` used by both model runner versions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.logger import logger
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
        self._burst_prefill_ids: set[str] = set()
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
        """Return the admission constraint for the next upstream schedule call."""
        running_requests = tuple(running_requests)
        pending_prefill_requests = tuple(pending_prefills)
        waiting_prefill_requests = tuple(waiting_prefills or ())
        pending_prefill_ids = frozenset(
            request.request_id for request in pending_prefill_requests
        )
        all_pending_requests: tuple[Request, ...] | None = None
        all_pending_ids: frozenset[str] | None = None

        def get_all_pending_requests() -> tuple[Request, ...]:
            nonlocal all_pending_requests
            if all_pending_requests is None:
                source = (
                    all_pending_prefills()
                    if all_pending_prefills is not None
                    else pending_prefill_requests
                )
                all_pending_requests = tuple(source)
            return all_pending_requests

        def get_all_pending_ids() -> frozenset[str]:
            nonlocal all_pending_ids
            if all_pending_ids is None:
                all_pending_ids = frozenset(
                    request.request_id for request in get_all_pending_requests()
                )
            return all_pending_ids

        def snapshot_prefill_ids() -> set[str]:
            running_prefill_ids = {
                request.request_id
                for request in running_requests
                if request.is_prefill_chunk
            }
            if max_prefill_slots is None:
                return set(get_all_pending_ids())
            waiting_prefill_ids = [
                request.request_id for request in waiting_prefill_requests
            ]
            waiting_capacity = max(
                0,
                max_prefill_slots - len(running_prefill_ids),
            )
            return running_prefill_ids | set(
                waiting_prefill_ids[:waiting_capacity]
            )

        if self._burst_prefill_ids:
            active_pending_ids = get_all_pending_ids()
            self._burst_prefill_ids.intersection_update(active_pending_ids)
            pending_prefill_ids = frozenset(self._burst_prefill_ids)
            if not self._burst_prefill_ids:
                pending_prefill_ids = active_pending_ids

        active_decodes = [
            request
            for request in running_requests
            if not request.is_prefill_chunk
            and self._decode_token_demand(request) > 0
        ]
        eligible_decodes = [
            request
            for request in active_decodes
            if scheduler_step >= request.next_decode_eligible_step
        ]
        active_decode_batch_size = len(active_decodes)
        decode_token_budget = min(
            max_token_budget,
            sum(self._decode_token_demand(request) for request in eligible_decodes),
        )

        running_full = (
            max_prefill_slots is not None
            and len(running_requests) >= max_prefill_slots
        )
        low_decode_release = (
            active_decode_batch_size < self.decode_low_watermark
        )

        # Low Decode always opens admission. Otherwise, reaching capacity
        # latches HOLD, which persists while running requests drain below
        # capacity until the low watermark is crossed.
        hold_released = self._prefill_hold and low_decode_release
        if low_decode_release:
            # if hold_released:
                # logger.info(
                #     "Prefill admission HOLD->OPEN: active_decodes=%d, "
                #     "decode_low_watermark=%d",
                #     active_decode_batch_size,
                #     self.decode_low_watermark,
                # )
            self._prefill_hold = False
        elif running_full and not self._prefill_hold:
            self._prefill_hold = True
            # logger.info(
            #     "Prefill admission OPEN->HOLD: running_requests=%d, "
            #     "max_prefill_slots=%d, active_decodes=%d",
            #     len(running_requests),
            #     max_prefill_slots,
            #     active_decode_batch_size,
            # )

        # State transitions still happen without pending Prefill work so future
        # arrivals observe the correct latched state. The current Decode-only
        # step does not need a constrained token budget in that case.
        if not pending_prefill_ids:
            return PrefillAdmissionDecision(
                False, None, "no_prefill", pending_prefill_ids
            )

        if self._prefill_hold:
            return PrefillAdmissionDecision(
                True,
                None,  # Running Prefill chunks retain the full token budget.
                "running_full_hold",
                pending_prefill_ids,
            )

        continuing_burst = bool(self._burst_prefill_ids)
        if not continuing_burst:
            self._burst_prefill_ids = snapshot_prefill_ids()
            pending_prefill_ids = frozenset(self._burst_prefill_ids)

        if hold_released:
            reason = "low_decode_release"
        elif continuing_burst:
            reason = "burst"
        else:
            reason = "open"

        # Running decode demand remains first in the token budget. Missing
        # decode slots approximate available PP bubbles; each burst reserves
        # one small Prefill quantum even at the watermark.
        bubble_capacity = max(
            self.decode_low_watermark - active_decode_batch_size,
            1,
        )
        prefill_token_budget = (
            bubble_capacity * self.prefill_tokens_per_pp_bubble
        )
        token_budget = min(
            max_token_budget, decode_token_budget + prefill_token_budget
        )
        return PrefillAdmissionDecision(
            False, token_budget, reason, pending_prefill_ids
        )

    def observe(
        self,
        decision: PrefillAdmissionDecision,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Compatibility hook; cohort progress is reconciled in ``decide``."""


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

    def _waiting_prefills(self) -> list[Request]:
        pending: dict[str, Request] = {}
        for request_queue in (self.waiting, self.skipped_waiting):
            for request in request_queue:
                if self._waiting_request_needs_prefill(request):
                    pending.setdefault(request.request_id, request)
        return list(pending.values())

    def _all_pending_prefills(self) -> list[Request]:
        pending: dict[str, Request] = {
            request.request_id: request for request in self.running if request.is_prefill_chunk
        }
        for request in self._waiting_prefills():
            pending.setdefault(request.request_id, request)
        return list(pending.values())

    def _defer_non_cohort_prefills(
        self, allowed_prefill_ids: frozenset[str]
    ) -> list[tuple[object, list[Request]]]:
        deferred: list[tuple[object, list[Request]]] = []
        for request_queue in (self.waiting, self.skipped_waiting):
            requests = [
                request
                for request in request_queue
                if (
                    self._waiting_request_needs_prefill(request)
                    and request.request_id not in allowed_prefill_ids
                )
            ]
            if requests:
                request_queue.remove_requests(requests)
                deferred.append((request_queue, requests))
        return deferred

    @staticmethod
    def _restore_deferred_prefills(
        deferred: list[tuple[object, list[Request]]],
    ) -> None:
        for request_queue, requests in deferred:
            for request in requests:
                request_queue.add_request(request)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        controller = self._prefill_admission_controller
        if controller is None:
            return super().schedule(throttle_prefills)

        num_waiting_for_streaming_input = getattr(
            self, "num_waiting_for_streaming_input", 0
        )

        max_prefill_slots = max(
            0,
            self.max_num_running_reqs - num_waiting_for_streaming_input,
        )
        decision = controller.decide(
            self.running,
            self._pending_prefills(),
            all_pending_prefills=self._all_pending_prefills,
            waiting_prefills=self._waiting_prefills(),
            max_prefill_slots=max_prefill_slots,
            scheduler_step=self.current_step + 1,
            max_token_budget=self.max_num_scheduled_tokens,
        )
        original_token_budget = self.max_num_scheduled_tokens

        if decision.token_budget is not None:
            self.max_num_scheduled_tokens = decision.token_budget

        deferred_prefills: list[tuple[object, list[Request]]] = []
        if decision.pending_prefill_ids:
            # HOLD excludes waiting Prefills, never already-running chunks.
            deferred_prefills = self._defer_non_cohort_prefills(
                frozenset() if decision.throttle_prefills else decision.pending_prefill_ids
            )

        try:
            scheduler_output = super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = original_token_budget
            self._restore_deferred_prefills(deferred_prefills)

        controller.observe(decision, scheduler_output)
        return scheduler_output


class PrefillAdmissionScheduler(_PrefillAdmissionSchedulerMixin, Scheduler):
    """Synchronous vLLM scheduler with prefill admission throttling."""


class PrefillAdmissionAsyncScheduler(_PrefillAdmissionSchedulerMixin, AsyncScheduler):
    """Asynchronous vLLM scheduler with prefill admission throttling."""
