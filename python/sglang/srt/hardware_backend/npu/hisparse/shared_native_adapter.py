"""Fused device adapters over unchanged HiSparse-owned plans and leases."""

import torch


def available(resident):
    return resident.device.type == "npu" and resident.native_lifecycle_available


def operations(resident):
    return resident._native_lifecycle


def reserve_inputs(owners, plan, logical):
    expected = torch.empty((owners.numel(), 9), dtype=torch.int64, device=owners.device)
    native_owners = torch.empty_like(owners)
    native_rows = torch.empty_like(plan.row_to_request)
    native_logical = torch.empty_like(logical)
    native_union = torch.empty_like(plan.union_index)
    return [expected, native_owners, native_rows, native_logical, native_union]


def complete(resident, lease, owners, expected, completion, status, ready, counts):
    if lease.resident is not resident or lease.layer_id not in resident._layer_index:
        raise ValueError("shared adapter completion requires this layer's lease")
    layer = resident.ensure_layer(lease.layer_id)
    success = torch.empty_like(lease.active)
    operations(resident).shared_adapter_complete(
        [
            owners,
            expected,
            completion,
            status,
            ready,
            counts,
            lease.active,
            lease.terminal,
            lease.serial,
            layer["execution_serial"],
            layer["in_flight"],
            layer["payload_valid"],
            layer["reusable"],
            resident.lifecycle_error,
        ],
        success,
        1,
    )
    resident.complete(
        lease.layer_id,
        lease.row_request_slots,
        lease.physical_keys,
        lease.slots,
        execution_mask=lease,
        success_mask=success,
    )
    return success
