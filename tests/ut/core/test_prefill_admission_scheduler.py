# SPDX-License-Identifier: Apache-2.0
"""HOLD must block admission without stalling an admitted prefill."""

from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.ascend_config import PrefillAdmissionConfig
from vllm_ascend.core.prefill_admission_scheduler import (
    PrefillAdmissionAsyncScheduler,
    PrefillAdmissionController,
    PrefillAdmissionScheduler,
)


@pytest.mark.parametrize("scheduler_cls", [PrefillAdmissionScheduler, PrefillAdmissionAsyncScheduler])
@pytest.mark.parametrize("fail", [False, True])
def test_hold_allows_running_chunks_without_hiding_waiting(scheduler_cls, fail):
    scheduler = object.__new__(scheduler_cls)
    scheduler.max_num_running_reqs = 16
    scheduler.max_num_scheduled_tokens = 4096
    scheduler.current_step = 0
    scheduler.prefill_capacity_bound = True
    scheduler._prefill_admission_controller = PrefillAdmissionController(
        PrefillAdmissionConfig(enabled=True, decode_low_watermark=10), 2
    )
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)

    def request(request_id, prefill=False, status=RequestStatus.RUNNING):
        return MagicMock(
            request_id=request_id,
            is_prefill_chunk=prefill,
            num_tokens=10000,
            num_tokens_with_spec=10000,
            num_computed_tokens=1000 if prefill else 10000,
            num_output_placeholders=0 if prefill else 1,
            next_decode_eligible_step=0,
            status=status,
        )

    admitted = request("admitted", prefill=True)
    scheduler.running = [request(str(i)) for i in range(15)] + [admitted]
    waiting = [request("new", True, RequestStatus.WAITING)]
    skipped = [request("resumed", True, RequestStatus.PREEMPTED)]
    scheduler.waiting.add_request(waiting[0])
    scheduler.skipped_waiting.add_request(skipped[0])

    def upstream(
        self, throttle_prefills=False, *,
        admit_new_prefills=True, separate_prefill_quota=False,
    ):
        assert not throttle_prefills
        assert self.max_num_scheduled_tokens == 4096
        assert self.prefill_capacity_bound
        assert not admit_new_prefills
        assert separate_prefill_quota
        assert list(self.waiting) == waiting
        assert list(self.skipped_waiting) == skipped
        assert admitted in self.running
        if fail:
            raise RuntimeError("upstream failure")
        admitted.num_computed_tokens = min(10000, admitted.num_computed_tokens + 3000)
        admitted.is_prefill_chunk = admitted.num_computed_tokens < 10000
        return MagicMock()

    with patch.object(Scheduler, "schedule", upstream):
        if fail:
            with pytest.raises(RuntimeError, match="upstream failure"):
                scheduler.schedule()
        else:
            for _ in range(3):
                scheduler.schedule()
            assert not admitted.is_prefill_chunk
    assert list(scheduler.waiting) == waiting
    assert list(scheduler.skipped_waiting) == skipped
    assert scheduler.max_num_scheduled_tokens == 4096
    assert scheduler.prefill_capacity_bound

    # HOLD survives draining to 10 decodes and opens strictly below 10.
    controller = scheduler._prefill_admission_controller
    for count, expected_hold in [(10, True), (9, False)]:
        decision = controller.decide(
            scheduler.running[:count], waiting,
            waiting_prefills=waiting, max_prefill_slots=16,
            scheduler_step=1, max_token_budget=4096,
        )
        assert decision.throttle_prefills is expected_hold


@pytest.mark.parametrize(
    "scheduler_cls", [PrefillAdmissionScheduler, PrefillAdmissionAsyncScheduler]
)
def test_prefill_burst_new_arrivals_remain_visible(scheduler_cls):
    scheduler = object.__new__(scheduler_cls)
    scheduler.max_num_running_reqs = 16
    scheduler.max_num_scheduled_tokens = 4096
    scheduler.current_step = 0
    scheduler._prefill_admission_controller = PrefillAdmissionController(
        PrefillAdmissionConfig(enabled=True, decode_low_watermark=8), 2
    )
    old = MagicMock(
        request_id="old", is_prefill_chunk=True,
        num_tokens=6000, num_computed_tokens=1000,
        status=RequestStatus.RUNNING,
    )
    new = MagicMock(
        request_id="new", is_prefill_chunk=False,
        num_tokens=1400, num_computed_tokens=0,
        status=RequestStatus.WAITING,
    )
    scheduler.running = [old]
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)

    def upstream(self, throttle_prefills=False, **kwargs):
        assert kwargs["admit_new_prefills"]
        assert kwargs["separate_prefill_quota"]
        return tuple(r.request_id for r in self.waiting)

    with patch.object(Scheduler, "schedule", upstream):
        assert scheduler.schedule() == ()
        scheduler.waiting.add_request(new)
        assert scheduler.schedule() == ("new",)
