"""Host ownership hooks shared by FSA capture and the NPU graph executor.

Capture discovers the bindings used by a graph. Only the outer replay owner can
record a fresh fence after all captured native reads and publication operations.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_CAPTURE_READERS = ContextVar("fsa_capture_readers", default=None)


class FsaGraphReaders:
    def __init__(self):
        self.readers = {}
        self.shared = {}
        self.retained = []
        self.states = {}
        self._acquired_guards = []
        self._replay_prepared = False
        self.capture_failed = False

    def _leases(self):
        seen = set()
        for retained in self.retained:
            if retained and id(retained[0]) not in seen:
                seen.add(id(retained[0]))
                yield retained[0]

    def _quarantine(self, guards, error):
        for guard in guards:
            if not any(value is self for value in guard.retained):
                guard.retained.append(self)
            guard.fail(error)
        for binding, _ in self.readers.values():
            binding.mark_reader_failure(error)

    def add(self, binding, epoch_input):
        self.readers[(id(binding), id(epoch_input))] = (binding, epoch_input)

    def add_shared(self, resident, guard, binding, epoch_input, retained, state=None):
        self.shared[id(guard)] = (resident, guard)
        self.retained.append(retained)
        if state is not None:
            self.states[id(state)] = state
        if binding is not None:
            self.add(binding, epoch_input)

    def finish_capture(self, device_module):
        if not self.readers and not self.shared:
            return
        try:
            # Capture may run on an internal stream; it is initialization, not
            # a replay reader fence. Drain before exposing the captured graph.
            from .shared_execution import drain_streams

            streams = []
            if self.shared:
                streams.append(device_module.current_stream())
            for _, guard in self.shared.values():
                main = getattr(guard, "main_stream", None)
                if main is not None and hasattr(main, "synchronize"):
                    streams.append(main)
                streams.extend(getattr(guard, "side_streams", ()))
            drain_streams(streams)
            device_module.synchronize()
            for lease in self._leases():
                submission = getattr(lease, "submission_token", None)
                if submission is not None:
                    if submission.item() and (
                        self.capture_failed or not lease._capture_complete
                    ):
                        raise RuntimeError(
                            "three-stage capture failed with uncertain lease initialization"
                        )
                    continue
                lease.resident.cancel_execution(
                    lease.layer_id, lease.row_request_slots, execution_mask=lease
                )
            # Cancel itself is device work; keep guards until it drains too.
            if self.shared:
                device_module.synchronize()
            for _, guard in self.shared.values():
                guard.drained()
        except Exception as error:
            self._quarantine((guard for _, guard in self.shared.values()), error)
            raise

    def before_replay(self):
        self._replay_prepared = False
        # Guard before epoch writes; all workspaces are acquired transactionally.
        acquired = self._acquired_guards = []
        try:
            if self.shared:
                import torch

                from .shared_execution import WorkspaceGuard

                stream = None
                for _, guard in self.shared.values():
                    if isinstance(guard, WorkspaceGuard) and guard.device.type == "npu":
                        if stream is None:
                            stream = torch.npu.current_stream()
                        guard.acquire(stream)
                    else:
                        guard.acquire()
                    acquired.append(guard)
        except Exception:
            # Outer replay owner drains any writes queued by earlier acquire/
            # refresh stages. Never expose a guard with pending writes.
            raise
        try:
            for state in self.states.values():
                state.refresh_generations()
            for binding, epoch_input in self.readers.values():
                binding.check_reader_health()
                if epoch_input is not None:
                    owners = [
                        state
                        for state in self.states.values()
                        if getattr(state, "three_stage_enabled", False)
                        and getattr(state, "mapping_epoch_input", None) is epoch_input
                    ]
                    if owners:
                        from .hicache_offload import fsa_resident_epoch

                        for state in owners:
                            state.refresh_mapping_epoch(fsa_resident_epoch(binding))
                    else:
                        epoch_input.fill_(binding.mapping_epoch)
            self._replay_prepared = True
        except Exception:
            # Outer replay owner drains any writes queued by earlier acquire/
            # refresh stages. Never expose a guard with pending writes.
            raise

    def after_replay(self, device_module):
        stream = None
        event = None
        shared_guards = {}
        candidates = [
            guard
            for _, guard in self.shared.values()
            if getattr(getattr(guard, "device", None), "type", None) == "npu"
            and callable(getattr(guard, "can_share_fence", None))
        ]
        if self.readers or candidates:
            stream = device_module.current_stream()
            shared_guards = {
                id(guard): guard
                for guard in candidates
                if guard.can_share_fence(stream)
            }
        if self.readers or shared_guards:
            event = device_module.Event()
            event.record(stream)
        if self.readers:
            bindings = {id(binding): binding for binding, _ in self.readers.values()}
            for binding in bindings.values():
                binding.record_reader(event)
        from .shared_execution import (
            _FORWARD,
            check_shared_residents,
            register_shared_resident,
        )

        for resident, guard in self.shared.values():
            if id(guard) in shared_guards:
                guard.release(fence=event, stream=stream)
            else:
                guard.release()
            register_shared_resident(resident)
        if self.shared and _FORWARD.get() is None:
            check_shared_residents(resident for resident, _ in self.shared.values())
        for lease in self._leases():
            if getattr(lease, "submission_token", None) is not None:
                lease._replay_initialized = True

    def drain_failure(self, device_module):
        if not self.readers and not self.shared:
            return
        try:
            from .shared_execution import drain_streams

            drain_streams(
                (
                    device_module.current_stream(),
                    *(
                        stream
                        for _, guard in self.shared.values()
                        for stream in getattr(guard, "side_streams", ())
                    ),
                )
            )
            owned = {
                (id(resident), getattr(guard, "layer_id", None))
                for resident, guard in self.shared.values()
                if guard in self._acquired_guards
            }
            for lease in self._leases():
                if (id(lease.resident), lease.layer_id) in owned:
                    submission = getattr(lease, "submission_token", None)
                    if submission is not None:
                        if not submission.item():
                            continue
                        if not lease._replay_initialized:
                            if not self._replay_prepared:
                                continue
                            raise RuntimeError(
                                "three-stage replay failed with uncertain lease initialization"
                            )
                    lease.resident.cancel_execution(
                        lease.layer_id, lease.row_request_slots, execution_mask=lease
                    )
            if self.shared:
                device_module.current_stream().synchronize()
            for guard in self._acquired_guards:
                guard.drained()
            self._acquired_guards = []
        except Exception as error:
            self._quarantine(self._acquired_guards, error)
            raise


@contextmanager
def collect_fsa_graph_readers():
    readers = FsaGraphReaders()
    token = _CAPTURE_READERS.set(readers)
    try:
        yield readers
    except BaseException:
        readers.capture_failed = True
        raise
    finally:
        _CAPTURE_READERS.reset(token)


def register_capture_reader(binding, epoch_input, state=None):
    readers = _CAPTURE_READERS.get()
    if readers is None:
        raise RuntimeError(
            "FSA Host capture requires NPUCudaGraphBackend replay ownership"
        )
    readers.add(binding, epoch_input)
    if state is not None:
        readers.states[id(state)] = state


@contextmanager
def fsa_host_reader(binding, state):
    if state.device.type != "npu":
        yield
        return

    import torch

    capturing = torch.npu.is_current_stream_capturing()
    binding.check_reader_health()
    if capturing:
        register_capture_reader(binding, None, state)
    try:
        yield
        if not capturing:
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
            binding.record_reader(event)
    except Exception:
        if not capturing:
            readers = FsaGraphReaders()
            readers.add(binding, None)
            readers.drain_failure(torch.npu)
        raise


def register_capture_shared(
    resident, guard, binding, epoch_input, retained, state=None
):
    readers = _CAPTURE_READERS.get()
    if readers is None:
        raise RuntimeError(
            "shared capture requires NPUCudaGraphBackend replay ownership"
        )
    readers.add_shared(resident, guard, binding, epoch_input, retained, state)
