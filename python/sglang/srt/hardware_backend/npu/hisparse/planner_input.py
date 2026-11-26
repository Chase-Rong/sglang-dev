"""Raw input layouts for three-stage HiSparse execution."""

from dataclasses import dataclass

import torch


@dataclass
class PlannerSelection:
    source: torch.Tensor
    query_lengths: torch.Tensor
    input_format: int
    pool_size: int
    real_rows: int
    padded_rows: int
    short_sequence_threshold: int
    raw_selection: torch.Tensor
    submitted: bool = False

    @property
    def width(self):
        return self.raw_selection.shape[1]

    @property
    def shape(self):
        return self.raw_selection.shape

    def mark_submitted(self):
        self.submitted = True


def _source(source, real_rows, padded_rows):
    if (
        source.ndim != 2
        or source.dtype != torch.int32
        or not 0 <= real_rows <= min(padded_rows, source.shape[0])
        or not 0 < padded_rows <= 6144
        or any(step < 0 for step in source.stride())
    ):
        raise ValueError("invalid Planner source shape, dtype, row capacity or stride")


def pooled_selection(
    source, query_lengths, pool_size, real_rows, padded_rows, short_threshold=0
):
    _source(source, real_rows, padded_rows)
    width = source.shape[1] * pool_size + pool_size - 1
    if (
        not isinstance(pool_size, int)
        or not 0 < pool_size <= 2147483647
        or not 0 < width <= 2051
        or query_lengths.ndim != 1
        or query_lengths.dtype != torch.int32
        or query_lengths.device != source.device
        or query_lengths.shape[0] < real_rows
        or query_lengths.stride(0) < 0
        or short_threshold not in (0, 2048)
        or (short_threshold and source.shape[1] * pool_size < 2048)
    ):
        raise ValueError("invalid pooled Planner geometry or per-query lengths")
    return PlannerSelection(
        source,
        query_lengths,
        0,
        pool_size,
        real_rows,
        padded_rows,
        short_threshold,
        torch.empty((padded_rows, width), dtype=torch.int32, device=source.device),
    )


def token_selection(source, padded_rows):
    if source.ndim == 3 and source.shape[1] == 1:
        source = source[:, 0]
    if source.dtype == torch.int64:
        valid = (source >= 0) & (source <= torch.iinfo(torch.int32).max)
        source = source.masked_fill(~valid, -1).to(dtype=torch.int32)
    _source(source, source.shape[0], padded_rows)
    if not 0 < source.shape[1] <= 2051:
        raise ValueError("invalid token Planner width")
    return PlannerSelection(
        source,
        torch.empty(0, dtype=torch.int32, device=source.device),
        1,
        1,
        source.shape[0],
        padded_rows,
        0,
        torch.empty(
            (padded_rows, source.shape[1]), dtype=torch.int32, device=source.device
        ),
    )


def completed_selection(selection):
    if isinstance(selection, PlannerSelection):
        if not selection.submitted:
            raise RuntimeError("raw Planner selection has not been submitted")
        return selection.raw_selection
    return selection
