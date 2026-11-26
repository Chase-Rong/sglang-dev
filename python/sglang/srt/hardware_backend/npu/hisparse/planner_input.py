"""Token IDs and output buffers for the three-stage external planner."""

from dataclasses import dataclass

import torch


@dataclass
class PlannerSelection:
    source: torch.Tensor
    query_lengths: torch.Tensor
    raw_selection: torch.Tensor


def token_selection(source, padded_rows):
    if source.ndim == 3 and source.shape[1] == 1:
        source = source[:, 0]
    if source.dtype == torch.int64:
        valid = (source >= 0) & (source <= torch.iinfo(torch.int32).max)
        source = source.masked_fill(~valid, -1).to(dtype=torch.int32)
    if (
        source.ndim != 2
        or source.dtype != torch.int32
        or not 0 <= source.shape[0] <= padded_rows <= 6144
        or not padded_rows
        or not 0 < source.shape[1] <= 2051
        or any(step < 0 for step in source.stride())
    ):
        raise ValueError("invalid external_plan token selection geometry")
    return PlannerSelection(
        source,
        source.new_empty(0),
        source.new_empty((padded_rows, source.shape[1])),
    )
