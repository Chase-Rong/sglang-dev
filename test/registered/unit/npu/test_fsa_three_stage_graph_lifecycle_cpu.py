import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.hisparse.fsa_graph_lifecycle import FsaGraphReaders
from sglang.srt.hardware_backend.npu.hisparse.shared_execution import WorkspaceGuard
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def setup_case(*, submitted=1, captured=True, legacy=False):
    calls = []

    class PoisonLease:
        @property
        def active(self):
            raise AssertionError("uninitialized lease active read")

        @property
        def terminal(self):
            raise AssertionError("uninitialized lease terminal read")

    lease = PoisonLease()
    lease.layer_id = 7
    lease.submission_token = (
        None if legacy else torch.tensor([submitted], dtype=torch.int32)
    )
    lease._capture_complete = captured
    lease._replay_initialized = False
    lease.row_request_slots = object()

    def cancel(*args, execution_mask):
        calls.append("cancel")
        if not legacy and not lease._replay_initialized:
            execution_mask.active

    resident = SimpleNamespace(
        cancel_execution=cancel,
        device=torch.device("cpu"),
        lifecycle_error=torch.tensor(0, dtype=torch.int32),
    )
    lease.resident = resident
    guard = WorkspaceGuard("cpu")
    guard.layer_id = 7
    guard.acquire()
    readers = FsaGraphReaders()
    payload = object()
    readers.add_shared(resident, guard, None, None, (lease, payload))
    # Real run_shared registers a lease once before submission and again with
    # the completed plan. Cleanup must tolerate both references.
    readers.add_shared(resident, guard, None, None, (lease,))
    stream = SimpleNamespace(synchronize=lambda: calls.append("drain"))
    device = SimpleNamespace(
        current_stream=lambda: stream, synchronize=lambda: calls.append("device-drain")
    )
    return readers, guard, lease, device, calls, payload


@pytest.mark.parametrize("submitted,captured", [(0, False), (1, True)])
def test_record_only_capture_never_reads_empty_lease(submitted, captured):
    readers, guard, lease, device, calls, _ = setup_case(
        submitted=submitted, captured=captured
    )
    readers.finish_capture(device)
    assert "cancel" not in calls
    assert not lease._replay_initialized
    assert not guard.active and guard.quarantine is None


def test_failed_capture_after_enqueue_quarantines_and_retains_without_cancel():
    readers, guard, _, device, calls, payload = setup_case(captured=False)
    with pytest.raises(RuntimeError, match="initialization"):
        readers.finish_capture(device)
    assert "cancel" not in calls
    assert guard.quarantine is not None
    assert readers in guard.retained
    assert any(payload in entry for entry in readers.retained)
    with pytest.raises(RuntimeError, match="quarantined"):
        guard.acquire()


def test_first_replay_failure_quarantines_without_reading_lease():
    readers, guard, _, device, calls, _ = setup_case()
    readers.finish_capture(device)
    readers.before_replay()
    with pytest.raises(RuntimeError, match="initialization"):
        readers.drain_failure(device)
    assert "cancel" not in calls and readers in guard.retained
    assert guard.quarantine is not None


def test_successful_replay_permits_later_drained_cancellation():
    readers, guard, lease, device, calls, _ = setup_case()
    readers.finish_capture(device)
    readers.before_replay()
    readers.after_replay(device)
    assert lease._replay_initialized
    readers.before_replay()
    readers.drain_failure(device)
    assert calls.count("cancel") == 1
    assert calls.index("drain") < calls.index("cancel")
    assert not guard.active and guard.quarantine is None


def test_refresh_failure_before_first_replay_skips_empty_lease():
    readers, guard, _, device, calls, _ = setup_case()
    readers.finish_capture(device)

    def fail_refresh():
        raise RuntimeError("refresh failed before replay")

    readers.states[1] = SimpleNamespace(refresh_generations=fail_refresh)
    with pytest.raises(RuntimeError, match="refresh failed"):
        readers.before_replay()
    readers.drain_failure(device)
    assert "cancel" not in calls
    assert not guard.active and guard.quarantine is None


def test_legacy_capture_still_cancels_initialized_lease():
    readers, guard, _, device, calls, _ = setup_case(legacy=True)
    readers.finish_capture(device)
    assert "cancel" in calls and not guard.active


@pytest.mark.parametrize("initialized", [False, True])
def test_failed_drain_keeps_lease_references_and_blocks_workspace(initialized):
    readers, guard, lease, device, calls, _ = setup_case()
    readers.finish_capture(device)
    lease._replay_initialized = initialized
    readers.before_replay()

    def fail_drain():
        raise RuntimeError("device drain failed")

    device.current_stream = lambda: SimpleNamespace(synchronize=fail_drain)
    with pytest.raises(RuntimeError, match="device drain failed"):
        readers.drain_failure(device)
    assert "cancel" not in calls
    assert readers in guard.retained
    with pytest.raises(RuntimeError, match="quarantined"):
        guard.acquire()


def test_actual_capture_collector_records_outer_failure():
    from sglang.srt.hardware_backend.npu.hisparse.fsa_graph_lifecycle import (
        collect_fsa_graph_readers,
    )

    with pytest.raises(ValueError, match="outside FSA"):
        with collect_fsa_graph_readers() as readers:
            raise ValueError("outside FSA")
    assert readers.capture_failed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
