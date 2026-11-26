"""Checked metadata sources retained by a native HiSparse execution."""

import torch


def available(state, workspace, logical, table, positions):
    resident = state.external_resident
    if (
        state.device.type != "npu"
        or not resident.native_lifecycle_available
        or hasattr(workspace, "warmup")
        or state.page_size != 128
        or not 0 < resident.max_routes <= 6
        or logical.dtype != torch.int32
        or logical.ndim != 3
        or logical.shape[1] != 1
        or not 0 < logical.shape[0] <= 6144
        or not 0 < logical.shape[2] <= 2051
        or not logical.is_contiguous()
        or table.dtype != torch.int32
        or table.ndim != 2
        or not 0 < table.shape[0] <= min(1024, logical.shape[0])
        or not 0 < table.shape[1] <= torch.iinfo(torch.int32).max // 128
        or table.stride(0) < 0
        or table.stride(1) <= 0
        or logical.device != state.device
        or table.device != logical.device
        or not 0 <= state.mutable_tail_tokens <= torch.iinfo(torch.uint32).max
    ):
        return False
    if positions is not None and (
        positions.device != logical.device
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.shape != (logical.shape[0],)
        or not positions.is_contiguous()
    ):
        return False
    return True


def reserve(state, workspace, logical, table, positions):
    rows, _, width = logical.shape
    if positions is None:
        positions = logical.new_empty(0, dtype=torch.int64)
    descriptor = (
        logical.view(rows, width),
        table,
        positions,
        state.generations,
        state.mapping_epoch_input,
        state.mutable_tail_tokens,
    )
    return (
        torch.empty_like(logical),
        logical.new_empty((rows, width), dtype=torch.int64),
        logical.new_empty((rows, width), dtype=torch.bool),
        torch.empty(table.shape, dtype=torch.int32, device=table.device),
        descriptor,
    )
