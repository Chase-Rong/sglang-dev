"""Bounded immutable union leases owned by the HiSparse resident manager."""

import torch


def reserve_union_execution(
    resident,
    layer_id,
    owners,
    keys,
    slots,
    cacheable,
    counts,
    *,
    planner_initializes=False,
):
    resident.ensure_layer(layer_id)
    if not resident.native_lifecycle_available:
        raise ValueError("union lifecycle requires a fixed HiSparse layer bank")
    if (
        keys.ndim != 2
        or not 0 < keys.shape[0] <= 1024
        or not 0 < keys.shape[1] <= 16384
        or keys.dtype != torch.int64
        or owners.shape != (keys.shape[0],)
        or owners.dtype != torch.int64
        or slots.shape != keys.shape
        or slots.dtype != torch.int32
        or cacheable.shape != keys.shape
        or cacheable.dtype != torch.bool
        or counts.shape != owners.shape
        or counts.dtype != torch.int32
        or any(
            value.device != resident.device
            for value in (owners, keys, slots, cacheable, counts)
        )
    ):
        raise ValueError("invalid resident union geometry, dtype or device")
    sources = tuple(
        value.contiguous() for value in (owners, keys, slots, cacheable, counts)
    )
    allocate_flags = torch.empty_like if planner_initializes else torch.zeros_like
    active = allocate_flags(resident.blocked_requests)
    submission = (
        torch.zeros(1, dtype=torch.int32, device="cpu") if planner_initializes else None
    )
    return resident._lease_type(
        resident,
        layer_id,
        active,
        torch.empty_like(resident.generation),
        torch.empty_like(resident.mapping_epoch),
        torch.empty_like(resident.planner_version),
        torch.empty_like(resident.generation),
        allocate_flags(active),
        torch.empty_like(owners),
        torch.empty_like(keys),
        torch.empty_like(keys),
        torch.empty_like(cacheable),
        native_ops=resident._native_lifecycle,
        native_inputs=sources,
        union_count=torch.empty_like(counts),
        submission_token=submission,
    )


def execute_union_attention(inputs, lease, adapters, host_address):
    resident = lease.resident
    state = resident.ensure_layer(lease.layer_id)
    native_inputs, outputs, context, cache = inputs.plan.native_execution
    metadata = getattr(inputs.plan, "builtin_metadata", None)
    selection = getattr(inputs.plan, "input_selection", None)
    if selection is None or metadata is None or inputs.plan.input_versions is None:
        raise ValueError("FSA requires raw Planner sources and version outputs")
    extra = (
        list(metadata[:5]),
        metadata[5],
        selection.query_lengths,
        *inputs.plan.input_versions,
        1,  # The indexer supplies expanded token IDs.
        1,  # Each entry selects one token.
        selection.source.shape[0],
        0,  # The indexer already handles short sequences.
        selection.raw_selection,
        lease.submission_token,
    )
    writeback = inputs.current_kv_writeback
    if writeback is not None:
        extra += (writeback.event_handle,)
    output = lease.native_ops.hisparse_fused_attention_three_stage(
        native_inputs,
        outputs,
        context,
        cache,
        [
            *(
                resident._bank_storage[name]
                for name in ("identity", "slot_allocated", "payload_valid", "reusable")
            ),
            resident.generation,
            resident.mapping_epoch,
            resident.planner_version,
            resident.release_state,
            resident.blocked_requests,
            state["in_flight"],
            state["execution_serial"],
            resident._in_flight,
            resident.lifecycle_error,
            resident._directory_storage["entries"],
            resident._directory_storage["control"],
        ],
        [
            lease.active,
            lease.generation,
            lease.mapping_epoch,
            lease.planner_version,
            lease.serial,
            lease.terminal,
            lease.row_request_slots,
            lease.physical_keys,
            lease.slots,
            lease.cacheable_mask,
            lease.union_count,
        ],
        adapters,
        inputs.query,
        inputs.workspace.payload,
        inputs.full_kv_cache,
        host_address,
        inputs.full_kv_block_table,
        inputs.workspace.native_workspace,
        resident.capacity,
        resident.max_routes,
        resident._layer_index[lease.layer_id],
        lease.layer_id,
        inputs.state.shared_instance,
        inputs.scale_value,
        1,
        *extra,
    )
    if writeback is not None:
        writeback.native_wait_queued = True
    lease._capture_complete = True
    return output


def cancel_submitted_lease(resident, layer_id, lease):
    submission = getattr(lease, "submission_token", None)
    if submission is not None and int(submission[0]) == 0:
        return
    resident.cancel_execution(layer_id, lease.row_request_slots, execution_mask=lease)
