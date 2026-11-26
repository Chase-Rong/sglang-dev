"""Device operations over HiSparse-owned resident banks and one-shot leases."""

import torch


def native_prepare_ops(device):
    if torch.device(device).type != "npu":
        return None
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops

    return native_ops()


def prepare_identity(resident, slots, generations, mapping_epoch):
    result = torch.empty_like(slots)
    resident._native_prepare.external_resident_prepare(
        [
            slots.contiguous(),
            generations.contiguous(),
            resident._epoch_tensor(mapping_epoch),
            *(
                resident._bank_storage[name]
                for name in ("identity", "slot_allocated", "payload_valid", "reusable")
            ),
            resident.generation,
            resident.mapping_epoch,
            resident.planner_version,
            resident.release_state,
            resident.blocked_requests,
            resident._in_flight,
            resident.lifecycle_error,
        ],
        result,
        resident.capacity,
        1,
    )
    return result


def native_lifecycle_ops(device):
    if torch.device(device).type != "npu":
        return None
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops

    return native_ops()


def reserve_execution(resident, layer_id, owners, keys, destinations, cacheable):
    resident.ensure_layer(layer_id)
    if not resident.native_lifecycle_available:
        raise ValueError("native lifecycle requires a fixed HiSparse layer bank")
    owners, keys, destinations = resident._plan_tensors(owners, keys, destinations)
    if keys.shape[0] <= 0 or keys.shape[1] <= 0:
        raise ValueError("native resident execution requires nonempty plan geometry")
    if cacheable is not None and (
        cacheable.shape != keys.shape or cacheable.dtype != torch.bool
    ):
        raise ValueError("execution cacheability must match physical keys")
    stable = torch.ones_like(keys, dtype=torch.bool) if cacheable is None else cacheable
    sources = tuple(
        value.contiguous() for value in (owners, keys, destinations, stable)
    )
    active = torch.zeros_like(resident.blocked_requests)
    return resident._lease_type(
        resident,
        layer_id,
        active,
        torch.empty_like(resident.generation),
        torch.empty_like(resident.mapping_epoch),
        torch.empty_like(resident.planner_version),
        torch.empty_like(resident.generation),
        torch.zeros_like(active),
        torch.empty_like(sources[0]),
        torch.empty_like(sources[1]),
        torch.empty_like(sources[2]),
        torch.empty_like(sources[3]),
        native_ops=resident._native_lifecycle,
        native_inputs=sources,
    )


def start_execution(resident, lease):
    if (
        lease.resident is not resident
        or lease.native_ops is not resident._native_lifecycle
        or lease.layer_id not in resident._layer_index
        or lease.native_inputs is None
    ):
        raise ValueError("native acquisition requires this resident's reserved lease")
    state = resident.ensure_layer(lease.layer_id)
    union = lease.union_count is not None
    acquire = (
        lease.native_ops.external_resident_union_acquire
        if union
        else lease.native_ops.external_resident_acquire
    )
    acquire(
        [
            *lease.native_inputs[:4],
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
            *([lease.native_inputs[4]] if union else []),
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
            *([lease.union_count] if union else []),
        ],
        resident.capacity,
        resident._layer_index[lease.layer_id],
        1,
    )


def complete_execution(resident, lease, success_mask, mapping_epoch):
    if (
        lease.resident is not resident
        or lease.native_ops is not resident._native_lifecycle
    ):
        raise ValueError("native completion requires this resident's execution lease")
    if success_mask is None:
        success = lease.active
    else:
        success = torch.as_tensor(
            success_mask, device=resident.device, dtype=torch.bool
        )
        if success.numel() == 1:
            success = success.expand_as(lease.active)
        if success.shape != lease.active.shape:
            raise ValueError(
                "native completion success mask must match request capacity"
            )
        success = success.contiguous()
    epoch = (
        lease.mapping_epoch[0]
        if mapping_epoch is None
        else resident._epoch_tensor(mapping_epoch)
    )
    state = resident.ensure_layer(lease.layer_id)
    union = lease.union_count is not None
    complete = (
        lease.native_ops.external_resident_union_complete
        if union
        else lease.native_ops.external_resident_complete
    )
    complete(
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
            success,
            epoch,
            resident.lifecycle_error,
            *([lease.union_count] if union else []),
        ],
        resident.capacity,
        resident._layer_index[lease.layer_id],
        mapping_epoch is not None,
        1,
    )
