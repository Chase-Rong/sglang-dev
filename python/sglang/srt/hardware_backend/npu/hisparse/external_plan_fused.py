"""ABI3 protected-hash planner and final consumer metadata.

Only validation-before-identity-adoption remains in the reservation adapter.
Planning and final consumer metadata are one independent SIMT kernel. Resident
record/lease/complete and Host reader ownership retain their existing lifecycle.
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

    def validate_inputs(
        self, physical_keys, query_ends, request_slots, *, cacheable_mask=None
    ):
        keys, ends, owners = self._input_tensors(
            physical_keys, query_ends, request_slots, cacheable_mask
        )
        rows = keys.shape[0]
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))
        malformed = ((ends < 0) | (ends > rows) | (ends < starts)).any()
        row = torch.arange(rows, dtype=torch.long, device=self.device)
        orphan = row >= ends[-1]
        malformed |= (orphan[:, None] & (keys != -1)).any()
        safe_ends = torch.cummax(ends.clamp(0, rows), dim=0).values
        lengths = safe_ends - torch.cat((safe_ends.new_zeros(1), safe_ends[:-1]))
        owner_valid = (owners > 0) & (owners < self.resident.request_capacity)
        safe_owners = owners.clamp(0, self.resident.request_capacity - 1)
        counts = torch.zeros(
            self.resident.request_capacity, dtype=torch.int32, device=self.device
        )
        counts.scatter_add_(
            0, safe_owners, (owner_valid & (lengths > 0)).to(torch.int32)
        )
        overflow = (
            malformed
            | (lengths > self.max_routes)
            | (owner_valid & (counts[safe_owners] > 1))
            | (owners < 0)
            | (owners >= self.resident.request_capacity)
        )
        request = torch.bucketize(row, safe_ends, right=True).clamp(
            max=owners.numel() - 1
        )
        key_invalid = (
            (row < safe_ends[-1])
            & owner_valid[request]
            & ((keys < -1) | (keys == torch.iinfo(torch.int64).max)).any(dim=1)
        )
        invalid_counts = torch.zeros_like(owners, dtype=torch.int32)
        invalid_counts.scatter_add_(0, request, key_invalid.to(torch.int32))
        overflow |= invalid_counts > 0
        self.resident._error(overflow, 2)
        return owners.masked_fill(~(owner_valid & (lengths > 0) & ~overflow), 0)

    def plan(
        self,
        layer_id,
        physical_keys,
        query_ends,
        request_slots,
        *,
        cacheable_mask=None,
        mapping_epoch=0,
        workspace=None,
        logical_topk=None,
        kv_lengths=None,
        host_tokens=None,
        defer_execution=False,
        input_selection=None,
    ):
        from sglang.srt.hardware_backend.npu.fsa_native import native_ops

        if input_selection is not None and not defer_execution:
            raise ValueError(
                "raw Planner input requires deferred three-stage execution"
            )
        if (
            workspace is None
            or logical_topk is None
            or kv_lengths is None
            or host_tokens is None
        ):
            raise ValueError("FUSED planner requires final SFA metadata reservation")
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
        if defer_execution:
            if self.device.type != "npu":
                raise ValueError("deferred fused execution requires the NPU producer")
        elif self.topology_cache is None:
            native_ops().external_plan_prepare(
                native_inputs,
                native_outputs,
                self.capacity,
                self.max_routes,
                int(host_tokens),
                3,
            )
        else:
            self.topology_ops.external_plan_prepare_cached(
                native_inputs,
                native_outputs,
                [
                    self.resident.generation,
                    self.resident.mapping_epoch,
                    self.resident.planner_version,
                ],
                list(self.topology_cache.tensors),
                self.capacity,
                self.max_routes,
                int(host_tokens),
                self.topology_cache.namespace,
                1,
            )
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
            mapping_epoch=(
                mapping_epoch
                if input_selection is not None
                else self.resident._epoch_tensor(mapping_epoch).clone()
            ),
            input_selection=input_selection,
            native_execution=(native_inputs, native_outputs, context, cache)
            if defer_execution
            else None,
        )
