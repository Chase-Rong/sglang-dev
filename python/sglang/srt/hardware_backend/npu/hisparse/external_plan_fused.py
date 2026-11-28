"""ABI3 protected-hash planner and final consumer metadata.

Prepare buffers for the device producer and fused V4 attention.
"""

import torch

from .external_plan_device import ExternalDeviceSelectionPlan, ExternalPlanReservation

PLANNER_CAPABILITIES = [3, 16384, 16384, 2051, 1024, 8, 216 * 1024, 193 * 1024, 1, 1, 1]


class ExternalPlanFusedPlanner(ExternalPlanReservation):
    def __init__(
        self, resident, rows, topk, batch_capacity, *, topology_ops=None, namespace=0
    ):
        if resident.capacity > 16384 or resident.max_routes > 6:
            raise ValueError("FUSED protected_hash requires C<=16384 and routes<=6")
        super().__init__(
            resident,
            rows,
            topk,
            batch_capacity,
            minimum_union_width=16,
        )
        if self.union_width > 16384:
            raise ValueError("FUSED protected_hash requires W<=16384")
        self.replacement_policy = "protected_hash"
        self.row_cacheable = torch.zeros_like(self.miss_mask)
        self.default_cacheable = torch.ones_like(self.miss_mask)
        self.row_owners = torch.zeros(rows, dtype=torch.int64, device=self.device)
        self.row_requests = torch.full(
            (rows,), -1, dtype=torch.int32, device=self.device
        )
        self.row_union = torch.full(
            (rows, topk), -1, dtype=torch.int32, device=self.device
        )
        self.overflow = torch.zeros(
            batch_capacity, dtype=torch.bool, device=self.device
        )
        self.topology_ops = topology_ops
        self.topology_cache = None
        if topology_ops is not None:
            from .external_plan_topology import ExternalPlanTopologyCache

            self.topology_cache = ExternalPlanTopologyCache(
                self.device, rows, topk, batch_capacity, self.union_width, namespace
            )

    def plan(
        self,
        layer_id,
        physical_keys,
        query_ends,
        request_slots,
        *,
        cacheable_mask,
        mapping_epoch,
        workspace,
        logical_topk,
        kv_lengths,
        input_selection,
    ):
        if self.device.type != "npu":
            raise ValueError("Deferred fused execution requires the NPU producer")
        rows, topk = physical_keys.shape
        batch = request_slots.numel()
        if not (
            0 < rows <= self.rows
            and topk == self.topk
            and 0 < batch <= self.batch_capacity
        ):
            raise ValueError("FUSED planner input exceeds fixed reservation")
        state = self.resident.ensure_layer(layer_id)
        stable = (
            self.default_cacheable[:rows] if cacheable_mask is None else cacheable_mask
        )
        native_inputs = [
            physical_keys,
            query_ends,
            request_slots,
            stable,
            logical_topk,
            kv_lengths,
            state["identity"],
            state["slot_allocated"],
            state["payload_valid"],
            state["reusable"],
            state["in_flight"],
            self.resident.blocked_requests,
            self.resident.release_state,
            self.resident.lifecycle_error,
        ]
        native_outputs = [
            self.slots[:rows],
            self.miss_mask[:rows],
            self.writer_mask[:rows],
            self.row_cacheable[:rows],
            self.row_owners[:rows],
            self.row_requests[:rows],
            self.row_union[:rows],
            self.union_keys[:batch],
            self.union_slots[:batch],
            self.union_owner[:batch],
            self.union_miss[:batch],
            self.union_cacheable[:batch],
            self.union_count[:batch],
            self.overflow[:batch],
            self.victims[:batch],
            workspace.ready[:batch],
            workspace.status[:batch],
            workspace.completion[:batch],
            workspace.counters[:batch],
            workspace.safe_topk[:rows],
            workspace.membership[:rows],
            workspace.selection_table[:rows],
            workspace.selection_status[:rows],
        ]
        context = [
            self.resident.generation,
            self.resident.mapping_epoch,
            self.resident.planner_version,
        ]
        cache = [] if self.topology_cache is None else list(self.topology_cache.tensors)
        return ExternalDeviceSelectionPlan(
            physical_keys=physical_keys,
            slots=self.slots[:rows],
            miss_mask=self.miss_mask[:rows],
            cacheable_mask=self.row_cacheable[:rows],
            writer_mask=self.writer_mask[:rows],
            row_request_slots=self.row_owners[:rows],
            union_keys=self.union_keys[:batch],
            union_slots=self.union_slots[:batch],
            union_count=self.union_count[:batch],
            layer_id=layer_id,
            row_to_request=self.row_requests[:rows],
            union_index=self.row_union[:rows],
            overflow=self.overflow[:batch],
            union_owner=self.union_owner[:batch],
            union_miss=self.union_miss[:batch],
            union_cacheable=self.union_cacheable[:batch],
            mapping_epoch=mapping_epoch,
            input_selection=input_selection,
            native_execution=(native_inputs, native_outputs, context, cache),
        )
