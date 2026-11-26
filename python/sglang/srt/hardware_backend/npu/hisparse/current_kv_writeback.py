"""Current-token Host writes overlapped with the metadata-only HiSparse Planner."""

import os
from collections import deque
from contextlib import contextmanager

import torch


def supports_current_kv_overlap():
    enabled = os.environ.get("SGLANG_NPU_FSA_CURRENT_KV_OVERLAP", "1")
    if enabled not in ("0", "1"):
        raise ValueError("SGLANG_NPU_FSA_CURRENT_KV_OVERLAP must be 0 or 1")
    return enabled == "1"


class CurrentKvWrite:
    def __init__(self, ready, main_stream=None):
        self.ready = ready
        self.main_stream = main_stream
        self.native_wait_queued = False
        self.completion = None

    @property
    def event_handle(self):
        handle = int(self.ready.npu_event)
        if handle <= 0:
            raise RuntimeError("current KV writeback event was not recorded")
        return handle


class CurrentKvWriteback:
    """Own a fixed copy stream and retain events through their consuming work."""

    def __init__(self, device, binding=None):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("current KV writeback must initialize before capture")
        self.stream = torch.npu.Stream(device=device)
        self.submitting = False
        self.retired = deque()
        self.capture_failures = []
        self.binding = binding
        self.failed = None

    def _reclaim(self):
        if len(self.retired) < 64:
            return
        while self.retired and self.retired[0].completion.query():
            self.retired.popleft()
        if len(self.retired) >= 128:
            self.retired[0].completion.synchronize()
            self.retired.popleft()

    @contextmanager
    def submit(self, write):
        if self.failed is not None:
            raise RuntimeError("current KV writeback is quarantined") from self.failed
        capturing = torch.npu.is_current_stream_capturing()
        if not capturing:
            self._reclaim()
        main = torch.npu.current_stream()
        self.stream.wait_stream(main)
        ticket = CurrentKvWrite(torch.npu.Event(), main)
        try:
            with torch.npu.stream(self.stream):
                self.submitting = True
                try:
                    write()
                finally:
                    self.submitting = False
                ticket.ready.record(self.stream)
            try:
                yield ticket
            finally:
                if not ticket.native_wait_queued:
                    main.wait_event(ticket.ready)
                if not capturing:
                    ticket.completion = torch.npu.Event()
                    ticket.completion.record(main)
                    self.retired.append(ticket)
        except BaseException as error:
            self.failed = error
            if self.binding is not None:
                self.binding.mark_reader_failure(error)
            self.capture_failures.append(ticket)
            if not capturing:
                self.drain()
            raise

    def drain(self):
        self.stream.synchronize()
        for ticket in self.retired:
            ticket.completion.synchronize()
        for ticket in self.capture_failures:
            if ticket.main_stream is not None:
                ticket.main_stream.synchronize()
        self.retired.clear()
        self.capture_failures.clear()
