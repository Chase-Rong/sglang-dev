"""Build raw Planner inputs without launching legacy metadata or identity work."""

import torch

from .planner_input import PlannerSelection, token_selection


def build_three_stage_inputs(state, **args):
    from .shared_execution import SharedInputs, drain_shared, reserve_shared

    q, full = args["q_nope"], args["full_kv"]
    rows = q.shape[0]
    selection = args["topk_indices"]
    if not isinstance(selection, PlannerSelection):
        selection = token_selection(selection, rows)
    if selection.padded_rows != rows or selection.source.device != q.device:
        raise ValueError("three-stage source differs from query bucket or device")
    table_input = args["block_table"]
    if (
        table_input.dtype not in (torch.int32, torch.int64)
        or table_input.device != q.device
        or table_input.ndim != 2
    ):
        raise ValueError(
            "three-stage requires an integer Host KV page table on query device"
        )
    if table_input.dtype == torch.int64:
        invalid = (table_input < -1) | (table_input > torch.iinfo(torch.int32).max)
        table_input = table_input.masked_fill(invalid, -2).to(dtype=torch.int32)
    ws = reserve_shared(
        state, args["layer_id"], rows, selection.width, args["full_rope"], full
    )
    ws.guard.acquire()
    capturing = state.device.type == "npu" and torch.npu.is_current_stream_capturing()
    try:
        resident = state.external_resident
        if capturing:
            from .fsa_graph_lifecycle import register_capture_shared

            register_capture_shared(
                resident, ws.guard, None, state.mapping_epoch_input, (), state
            )
        else:
            state.refresh_generations()
            state.refresh_mapping_epoch(args["mapping_epoch"])
        ends = (
            args["query_lengths"]
            .to(device=state.device, dtype=torch.int32)
            .contiguous()
        )
        lengths = (
            args["kv_lengths"].to(device=state.device, dtype=torch.int32).contiguous()
        )
        owners = (
            args["request_slots"]
            .to(device=state.device, dtype=torch.int64)
            .contiguous()
        )
        batch = owners.numel()
        if (
            not 0 < batch <= ws.planner.batch_capacity
            or ends.numel() != batch
            or lengths.numel() != batch
            or table_input.shape[0] != batch
        ):
            raise ValueError(
                "three-stage request vectors/page table exceed reservation"
            )
        positions = args["query_positions"]
        if positions is None:
            positions = ws.empty_positions
        elif (
            positions.device != q.device
            or positions.dtype not in (torch.int32, torch.int64)
            or positions.ndim != 1
            or positions.numel() != rows
            or not positions.is_contiguous()
        ):
            raise ValueError("three-stage positions must match the query bucket")
        table = torch.empty(table_input.shape, dtype=torch.int32, device=q.device)
        logical, physical, stable = (
            ws.input_logical[:rows],
            ws.input_physical[:rows],
            ws.input_stable[:rows],
        )
        plan = ws.planner.plan(
            args["layer_id"],
            physical,
            ends,
            owners,
            cacheable_mask=stable,
            mapping_epoch=state.mapping_epoch_input,
            workspace=ws,
            logical_topk=logical,
            kv_lengths=lengths,
            host_tokens=full.shape[0] * 128,
            defer_execution=True,
            input_selection=selection,
        )
        plan.builtin_metadata = (
            selection.source,
            table_input,
            positions,
            state.generations,
            state.mapping_epoch_input,
            state.mutable_tail_tokens,
        )
        plan.input_versions = (ws.prepared_owners[:batch], ws.gathered_versions[:batch])
        return SharedInputs(
            q,
            full,
            table,
            lengths,
            ends,
            selection.raw_selection[:, None],
            owners,
            plan,
            ws,
            state,
            args["layer_id"],
            args["scale"],
            state.mapping_epoch_input,
            load_mode=state.load_mode,
        )
    except Exception:
        if not capturing:
            drain_shared(ws, None)
            ws.guard.drained()
        raise
