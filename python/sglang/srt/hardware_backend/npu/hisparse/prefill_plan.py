"""Bounded CPU planning for admitted integrated Prefill chunks."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.hardware_backend.npu.prefill_dma import mapped_span_runs


def build_prefill_page_plans(
    ordered_pages, prefix_lens, seq_lens, page_table, page_size, capacity
):
    """Build DMA spans from ordered physical pages and valid per-request ranges."""
    ordered_pages = ordered_pages.numpy()
    page_table = page_table.numpy()
    prefix_sizes = np.zeros_like(page_table)
    new_pages = []
    new_offsets = []
    new_sizes = []
    start = 0
    for prefix_len, seq_len in zip(prefix_lens, seq_lens):
        page_count = (seq_len + page_size - 1) // page_size
        end = start + page_count
        pages = ordered_pages[start:end]
        logical_starts = np.arange(page_count) * page_size
        prefix_counts = np.clip(prefix_len - logical_starts, 0, page_size)
        np.maximum.at(prefix_sizes, pages, prefix_counts)
        if prefix_len < seq_len:
            first_page = prefix_len // page_size
            starts = logical_starts[first_page:]
            offsets = np.maximum(prefix_len - starts, 0)
            new_pages.append(pages[first_page:])
            new_offsets.append(offsets)
            new_sizes.append(np.minimum(seq_len - starts, page_size) - offsets)
        start = end

    device_capacity = page_table.size * page_size
    prefix_pages = np.flatnonzero(prefix_sizes)
    prefix_plan = mapped_span_runs(
        page_table[prefix_pages] * page_size,
        prefix_pages * page_size,
        prefix_sizes[prefix_pages],
        capacity,
        device_capacity,
    )
    if not new_pages:
        return prefix_plan, torch.empty((0, 3), dtype=torch.int64)
    pages = np.concatenate(new_pages)
    offsets = np.concatenate(new_offsets)
    return prefix_plan, mapped_span_runs(
        pages * page_size + offsets,
        page_table[pages] * page_size + offsets,
        np.concatenate(new_sizes),
        device_capacity,
        capacity,
    )


@dataclass(frozen=True)
class _PendingPlan:
    write_locations: torch.Tensor
    request_indices: torch.Tensor
    slots: tuple
    prefix_lens: tuple
    seq_lens: tuple
    request_generations: torch.Tensor
    device_pages: torch.Tensor
    host_pages: torch.Tensor
    host_generations: torch.Tensor
    future: Future


class PrefillChunkPlanner:
    """Plans retain CPU metadata only, with no lease on a discarded batch."""

    def __init__(self, backing, request_pool):
        self.backing = backing
        self.request_pool = request_pool
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefill")
        self._inflight = None
        self._pending = None

    def discard(self):
        if self._pending is not None:
            self._pending.future.cancel()
            self._pending = None

    def close(self):
        self.discard()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def submit(
        self,
        ordered_pages,
        write_locations,
        request_indices,
        slots,
        prefix_lens,
        seq_lens,
    ):
        self.discard()
        if self._inflight is not None and not self._inflight.done():
            return
        if not slots.numel():
            return
        prefix_lens = tuple(prefix_lens)
        seq_lens = tuple(seq_lens.tolist())
        page_size = self.backing.page_size
        if (
            ordered_pages.numel()
            != sum((length + page_size - 1) // page_size for length in seq_lens)
            or len(prefix_lens) != len(seq_lens)
            or len(prefix_lens) != slots.numel()
            or any(
                not 0 <= prefix <= length
                for prefix, length in zip(prefix_lens, seq_lens)
            )
            or write_locations.numel()
            != sum(length - prefix for prefix, length in zip(prefix_lens, seq_lens))
        ):
            return
        binding = self.backing.page_binding
        pages = ordered_pages.unique()
        with binding.lock:
            binding.check_reader_health()
            binding._validate_device_pages(pages)
            page_table = binding.device_to_host_cpu.clone()
            host_pages = page_table[pages]
            if (host_pages <= 0).any():
                raise RuntimeError("Prefill has unmapped Host KV pages")
            host_generations = binding.host_generations[host_pages]
        request_generations = self.request_pool.req_generation[slots]
        self._inflight = self._executor.submit(
            build_prefill_page_plans,
            ordered_pages,
            prefix_lens,
            seq_lens,
            page_table,
            page_size,
            self.backing.pages * page_size,
        )
        self._pending = _PendingPlan(
            write_locations,
            request_indices,
            tuple(slots.tolist()),
            prefix_lens,
            seq_lens,
            request_generations,
            pages,
            host_pages,
            host_generations,
            self._inflight,
        )

    def take(self, batch, slots):
        pending = self._pending
        if pending is None:
            return None
        if (
            batch.out_cache_loc is not pending.write_locations
            or batch.req_pool_indices is not pending.request_indices
            or tuple(slots.tolist()) != pending.slots
            or tuple(batch.extend_prefix_lens_cpu) != pending.prefix_lens
            or batch.seq_lens_cpu is None
            or tuple(batch.seq_lens_cpu.tolist()) != pending.seq_lens
            or not torch.equal(
                self.request_pool.req_generation[slots], pending.request_generations
            )
        ):
            self.discard()
            return None
        binding = self.backing.page_binding
        with binding.lock:
            binding.check_reader_health()
            current = binding.device_to_host_cpu[pending.device_pages]
            if not torch.equal(current, pending.host_pages) or not torch.equal(
                binding.host_generations[current], pending.host_generations
            ):
                self.discard()
                return None
        self._pending = None
        return pending.future.result()
