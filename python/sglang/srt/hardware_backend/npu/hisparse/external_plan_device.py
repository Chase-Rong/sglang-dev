"""Independent device planning contract for ``external_plan``.

Physical Host token keys, rather than logical TopK IDs, are deduplicated per
request.  All operations retain fixed shapes and stay on the input device;
there are no registered Host buffers or Host planner callbacks.
"""

from dataclasses import dataclass

import torch


@dataclass
class ExternalDeviceSelectionPlan:
    physical_keys: torch.Tensor
    slots: torch.Tensor
    miss_mask: torch.Tensor
    cacheable_mask: torch.Tensor | None
    writer_mask: torch.Tensor
    row_request_slots: torch.Tensor
    union_keys: torch.Tensor
    union_slots: torch.Tensor
    union_count: torch.Tensor
    layer_id: int
    row_to_request: torch.Tensor
    union_index: torch.Tensor
    overflow: torch.Tensor
    union_owner: torch.Tensor
    union_miss: torch.Tensor
    union_cacheable: torch.Tensor
    mapping_epoch: int = 0
    request_union: bool = True
    committed: bool = False
    native_execution: tuple | None = None
    builtin_metadata: tuple | None = None
    input_selection: object = None
    input_versions: tuple | None = None


class ExternalPlanReservation:
    """Fixed buffers for the external-plan producer."""

    request_union = True

    def __init__(
        self,
        resident,
        rows,
        topk,
        batch_capacity,
        *,
        minimum_union_width=16,
    ):
        if min(rows, topk, batch_capacity) <= 0 or (topk > 2051):
            raise ValueError("invalid external device planner geometry")
        if rows > batch_capacity * resident.max_routes:
            raise ValueError("query rows exceed reserved request routes")
        if resident.capacity < resident.max_routes * topk:
            raise ValueError("external device capacity must cover request union")
        if resident.capacity > 32767 or resident.request_capacity > 32767:
            raise ValueError("external device planner exceeds the int16 ABI")
        self.resident = resident
        self.device = resident.device
        self.rows = rows
        self.topk = topk
        self.capacity = resident.capacity
        self.batch_capacity = batch_capacity
        self.max_routes = resident.max_routes
        self.union_width = max(
            minimum_union_width, 1 << (self.max_routes * topk - 1).bit_length()
        )
        self.union_keys = torch.full(
            (batch_capacity, self.union_width),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.union_slots = torch.full_like(self.union_keys, -1, dtype=torch.int32)
        self.union_owner = torch.full_like(self.union_slots, -1)
        self.union_miss = torch.zeros_like(self.union_slots, dtype=torch.bool)
        self.union_cacheable = torch.zeros_like(self.union_miss)
        self.union_count = torch.zeros(
            batch_capacity, dtype=torch.int32, device=self.device
        )
        self.victims = torch.full(
            (batch_capacity, self.capacity), -1, dtype=torch.int64, device=self.device
        )
        self.slots = torch.full((rows, topk), -1, dtype=torch.int64, device=self.device)
        self.miss_mask = torch.zeros_like(self.slots, dtype=torch.bool)
        self.writer_mask = torch.zeros_like(self.miss_mask)
