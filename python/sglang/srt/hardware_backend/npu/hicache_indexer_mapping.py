"""CPU page lists for NPU HiCache transfers and continuing Prefill."""

import torch


def prepare_npu_hicache_indices(host_indices, device_indices, io_backend):
    if io_backend not in ("kernel", "direct", "kernel_ascend"):
        raise ValueError(f"Unsupported NPU HiCache IO backend: {io_backend}")
    for indices in (host_indices, device_indices):
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("NPU HiCache indices must be integer vectors")
    return tuple(
        indices.detach().to(device="cpu", dtype=torch.int64).clone()
        for indices in (host_indices, device_indices)
    )


def get_indexer_mapping(allocator):
    """Resident copies must finish before reusing their source pages."""
    allocator = getattr(allocator, "full_attn_allocator", allocator)
    pool = allocator.get_kvcache()
    pool = getattr(pool, "full_kv_pool", pool)
    return getattr(pool, "hicache_indexer_mapping", None)


def get_prefill_device_pages(batch, req_pool_indices_cpu, seq_lens_cpu):
    """Read each request's allocator pages after the scheduler installs its rows."""
    table = batch.req_to_token_pool.req_to_token
    page_size = batch.tree_cache.token_to_kv_pool_allocator.page_size
    counts = (seq_lens_cpu + page_size - 1) // page_size
    requests = req_pool_indices_cpu.repeat_interleave(counts)
    starts = (counts.cumsum(0) - counts).repeat_interleave(counts)
    columns = (torch.arange(requests.numel()) - starts) * page_size
    rows = table[requests.to(table.device), columns.to(table.device)].to(
        "cpu", torch.int64
    )
    if (rows <= 0).any() or (rows % page_size).any():
        raise ValueError("Prefill page table must contain allocated, aligned pages")
    return rows // page_size
