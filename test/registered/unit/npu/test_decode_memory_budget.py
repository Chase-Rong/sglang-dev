import sys
from dataclasses import replace

import pytest

from sglang.srt.hardware_backend.npu.hisparse import decode_memory_budget as MODULE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def budget(**overrides):
    values = dict(
        execution_abi="shared_v3",
        layers=4,
        latent_dim=512,
        rope_dim=0,
        index_dim=128,
        element_bytes=2,
        topk=4,
        resident_capacity=128,
        host_bytes=1 << 34,
    )
    values.update(overrides)
    return MODULE.DecodeKvMemoryBudget(**values)


@pytest.mark.parametrize(
    "requests,graph_rows,routes", [(1, 0, 1), (10, 0, 6), (3, 10, 6)]
)
@pytest.mark.parametrize("layers", [1, 5])
def test_fixed_workspace_charges_per_physical_layer(
    requests, graph_rows, routes, layers
):
    model = budget(
        layers=layers,
        graph_rows=graph_rows,
        rows_per_request=routes,
        native_workspace_bytes=4096,
    )
    baseline = MODULE.shared_storage_bytes(
        model.external_request_slots(requests),
        max(requests, graph_rows),
        model.row_capacity(requests),
        layers,
        model.topk,
        model.selection_capacity(128),
        128,
        routes,
        native_workspace_bytes=4096,
    )
    actual = model.shared_persistent_bytes(requests, 128)
    assert actual["cpu"] == baseline["cpu_generation_snapshot"]
    charged_workspace = (
        actual["device"] - baseline["device_owner_global"] - baseline["device_layers"]
    )
    assert charged_workspace == 0
    without_workspace = replace(model, native_workspace_bytes=0)
    assert (
        actual["device"]
        - without_workspace.shared_persistent_bytes(requests, 128)["device"]
    ) == layers * 4096


def test_shared_workspace_is_reused_by_graph_buckets_and_steps():
    model = budget(graph_rows=10, rows_per_request=6, callback_instances=1)
    captures = replace(model, callback_instances=7)
    assert captures.shared_persistent_bytes(2, 128) == model.shared_persistent_bytes(
        2, 128
    )
    retained_snapshot = MODULE.shared_capture_storage_bytes(
        model.external_request_slots(2),
        10,
        60,
        model.topk,
        union_width=1 << (model.rows_per_request * model.topk - 1).bit_length(),
        cacheable_mask=True,
        builtin_metadata=True,
    )
    assert captures.fixed_bytes(2, 128)[1] - model.fixed_bytes(2, 128)[1] == (
        6 * model.layers * retained_snapshot
    )


def test_small_topk_graph_reserves_native_minimum_union_width():
    model = budget(topk=1, graph_rows=2)
    snapshots = replace(model, callback_instances=2)
    expected = MODULE.shared_capture_storage_bytes(
        model.external_request_slots(2),
        2,
        2,
        1,
        union_width=16,
        cacheable_mask=True,
        builtin_metadata=True,
    )
    assert (
        snapshots.fixed_bytes(2, 128)[1] - model.fixed_bytes(2, 128)[1]
        == model.layers * expected
    )


def test_workspace_budget_uses_attention_workspace_without_legacy_finish_reserve():
    assert budget().native_workspace_bytes == 30932992


@pytest.mark.parametrize("routes,width", [(1, 4096), (6, 16384)])
@pytest.mark.parametrize("cacheable_mask", [False, True])
def test_union_capture_budget_charges_worst_case_snapshot(
    routes, width, cacheable_mask
):
    requests, batches, topk = 11, 10, 2051
    rows = batches * routes
    actual = MODULE.shared_capture_storage_bytes(
        requests, batches, rows, topk, union_width=width, cacheable_mask=cacheable_mask
    )
    expected = (
        35 * requests
        + (20 + int(cacheable_mask)) * rows * topk
        + 4 * rows
        + 17 * batches * width
        + 92 * batches
        + 8
    )
    assert actual == expected


@pytest.mark.parametrize("width", [0, -1, 16385])
def test_union_capture_budget_rejects_unsupported_width(width):
    with pytest.raises(ValueError, match="union lease width"):
        MODULE.shared_capture_storage_bytes(11, 10, 60, 2051, union_width=width)


def test_shared_storage_rejects_row_capacity_above_native_limit():
    with pytest.raises(ValueError, match="geometry"):
        MODULE.shared_storage_bytes(5, 2, 6145, 1, 4, 128, 128, 2)


def test_joint_solver_reprices_shared_storage_when_reducing_requests():
    model = budget(layers=2, rows_per_request=6, native_workspace_bytes=4096)
    available = model.required_bytes(3 * 128, 3, 128)[1]
    result = MODULE.solve_joint_dsa_kda_tokens(
        budget=model,
        available_bytes=available,
        token_limit=1024,
        page_size=128,
        requests=4,
        state_bytes_per_slot=0,
    )
    assert result.requests == 3
    assert result.target_tokens == 3 * 128
    assert result.dsa_bytes == available
    assert result.unused_bytes == 0


def test_mtp_sizes_query_rows_instead_of_request_count():
    ordinary = budget()
    mtp = budget(rows_per_request=4)
    assert mtp.row_capacity(2) == 8
    mtp_host, mtp_device = mtp.fixed_bytes(2, 128)
    ordinary_host, ordinary_device = ordinary.fixed_bytes(8, 128)
    assert mtp_host < ordinary_host  # CPU generation snapshots are request-owned.
    # Row payload storage stays proportional to query rows, while device
    # planner resident-sort workspaces are allocated per active request.
    assert mtp_device < ordinary_device
    graph = budget(rows_per_request=4, graph_rows=8)
    assert graph.row_capacity(1) == 32
    assert graph.row_capacity(9) == 36


@pytest.mark.parametrize("graph_rows,routes,batch", [(0, 1, 4), (0, 3, 4), (8, 1, 8)])
def test_eager_padding_reserves_workspace_for_padded_batches(graph_rows, routes, batch):
    requests = 2
    model = budget(
        graph_rows=graph_rows,
        rows_per_request=routes,
        pad_eager_batch_size=lambda size: (size + 3) // 4 * 4,
    )
    assert model.external_request_slots(requests) == 3
    assert model.row_capacity(requests) == batch * routes
    expected = MODULE.shared_storage_bytes(
        3,
        batch,
        batch * routes,
        model.layers,
        model.topk,
        model.selection_capacity(128),
        128,
        routes,
        native_workspace_bytes=model.native_workspace_bytes,
    )
    assert model.shared_persistent_bytes(requests, 128)["device"] == (
        expected["device_owner_global"] + expected["device_layers"]
    )


def test_target_and_draft_budget_includes_both_backings_and_selection():
    target = budget(layers=4)
    combined = budget(layers=5, rows_per_request=4)
    assert combined.host_bytes_per_token == target.host_bytes_per_token * 5 // 4
    assert combined.device_bytes_per_token == target.device_bytes_per_token * 5 // 4
    assert combined.fixed_bytes(2, 128)[1] > target.fixed_bytes(2, 128)[1]


def test_full_kv_scales_host_while_indexer_scales_npu():
    model = budget()
    before = model.required_bytes(8192, 4, 128)
    after = model.required_bytes(16384, 4, 128)
    assert after[0] - before[0] == 8192 * 4 * 512 * 2
    assert after[1] - before[1] == 8192 * 4 * 128 * 2


def test_independent_host_capacity_does_not_consume_device_kv_budget():
    ordinary = budget(prefill_layer_slots=2)
    expanded = budget(prefill_layer_slots=2, host_token_capacity=131072)
    before = ordinary.required_bytes(8192, 4, 128)
    after = expanded.required_bytes(8192, 4, 128)
    assert after[0] - before[0] == 122880 * 4 * 512 * 2
    assert after[1] == before[1]
    assert expanded.max_tokens(before[1], 128, lambda _: 4) == 8192
    # This is a minimum Host capacity, not a cap on the device solver.
    assert expanded.required_bytes(262144, 4, 128) == ordinary.required_bytes(
        262144, 4, 128
    )


def test_independent_host_budget_includes_target_draft_and_reserved_page():
    model = budget(layers=5, host_token_capacity=131072)
    host_fixed, _ = model.fixed_bytes(4, 128)
    assert model.required_bytes(8192, 4, 128)[0] == host_fixed + 131200 * 5 * 512 * 2
    required, device_bytes = model.required_bytes(8192, 4, 128)
    too_small = replace(model, host_bytes=required - 1)
    with pytest.raises(MemoryError, match="Host"):
        too_small.validate(8192, 4, 128, device_bytes)
    assert too_small.max_tokens(device_bytes, 128, lambda _: 4) == 0


def test_independent_host_capacity_requires_nonnegative_aligned_tokens():
    with pytest.raises(ValueError, match="Host"):
        budget(host_token_capacity=-1)
    with pytest.raises(ValueError, match="Host"):
        budget(host_token_capacity=129).required_bytes(8192, 4, 128)


def test_integrated_slots_add_device_capacity_without_duplicating_host():
    ordinary = budget()
    integrated = budget(prefill_layer_slots=2)
    before = ordinary.required_bytes(8192, 1, 128)
    after = integrated.required_bytes(8192, 1, 128)
    assert after[0] == before[0]
    assert after[1] - before[1] == (8192 + 128) * 2 * 512 * 2
    assert integrated.max_tokens(after[1], 128, lambda tokens: 1) == 8192


def test_staging_reserves_one_fixed_window_and_copy_metadata():
    baseline = budget()
    staged = budget(staging_bytes=4096)
    for rows in (1, 8):
        before = baseline.required_bytes(8192, rows, 128)
        after = staged.required_bytes(8192, rows, 128)
        assert after[0] == before[0]
        assert after[1] - before[1] == 4096
    with pytest.raises(ValueError, match="staging"):
        budget(staging_bytes=33)


def test_solver_reserves_padding_page_and_fixed_selection_storage():
    model = budget()
    _, device_bytes = model.required_bytes(8192, 4, 128)
    assert model.max_tokens(device_bytes, 128, lambda tokens: 4) == 8192
    assert model.max_tokens(device_bytes - 1, 128, lambda tokens: 4) == 8064


def test_host_limit_includes_planner_and_padding():
    model = budget()
    host_required, device_required = model.required_bytes(8192, 4, 128)
    constrained = replace(model, host_bytes=host_required)
    assert constrained.max_tokens(device_required * 100, 128, lambda tokens: 4) == 8192
    with pytest.raises(MemoryError, match="Host"):
        constrained.validate(8320, 4, 128)


def test_graph_reservation_exceeds_real_request_count():
    model = budget(graph_rows=8)
    assert model.fixed_bytes(1, 128)[0] < model.fixed_bytes(8, 128)[0]
    assert model.row_capacity(1) == model.row_capacity(8)
    assert model.fixed_bytes(1, 128)[1] < model.fixed_bytes(8, 128)[1]
    assert model.fixed_bytes(9, 128)[1] > model.fixed_bytes(8, 128)[1]
    assert model.fixed_bytes(1, 128)[1] > budget().fixed_bytes(1, 128)[1]


def test_joint_solver_handles_request_count_threshold():
    model = budget()

    def row_count(tokens):
        return 2 if tokens <= 8192 else 4

    _, device_bytes = model.required_bytes(8192, 2, 128)
    result = model.max_tokens(device_bytes, 128, row_count)
    assert result == 8192
    model.validate(result, row_count(result), 128, device_bytes)
    with pytest.raises(MemoryError, match="Indexer"):
        model.validate(result + 128, row_count(result + 128), 128, device_bytes)


@pytest.mark.parametrize("page_size", [0, -1])
def test_invalid_page_size_rejected(page_size):
    with pytest.raises(ValueError):
        budget().required_bytes(8192, 4, page_size)


def test_alignment_cannot_overflow_native_slot_encoding():
    with pytest.raises(ValueError, match="C<=16384"):
        budget(resident_capacity=32767).fixed_bytes(4, 128)


def test_insufficient_npu_space_returns_no_capacity():
    _, fixed = budget().fixed_bytes(4, 128)
    assert budget().max_tokens(fixed, 128, lambda tokens: 4) == 0


def solve_request_priority(**overrides):
    values = dict(
        budget=budget(prefill_layer_slots=2),
        available_bytes=1 << 30,
        token_limit=1048576,
        page_size=128,
        requests=8,
        state_bytes_per_slot=1 << 20,
        request_bytes_per_row=32768,
    )
    values.update(overrides)
    return MODULE.solve_joint_dsa_kda_tokens(**values)


def test_request_priority_shrinks_tokens_before_concurrency():
    result = solve_request_priority()
    assert result.requests == result.state_slots == 8
    assert 8 * 128 <= result.target_tokens < 1048576
    assert result.target_tokens % 128 == 0
    assert (
        result.unused_bytes < 128 * budget(prefill_layer_slots=2).device_bytes_per_token
    )
    assert (
        result.dsa_bytes + result.kda_bytes + result.request_bytes + result.unused_bytes
        == 1 << 30
    )


def test_request_priority_more_memory_increases_tokens_when_requests_fit():
    small = solve_request_priority(available_bytes=1 << 29)
    large = solve_request_priority()
    assert small.requests == large.requests == 8
    assert small.state_slots == large.state_slots == 8
    assert small.target_tokens < large.target_tokens


def test_request_priority_token_cap_is_shared_and_not_multiplied_or_divided():
    result = solve_request_priority(token_limit=8193)
    assert result.target_tokens == 8192
    assert result.requests == 8
    assert result.unused_bytes > 0


def test_request_priority_host_limit_also_reduces_tokens():
    model = budget(prefill_layer_slots=2)
    host_bytes = model.required_bytes(8192, 8, 128)[0]
    result = solve_request_priority(budget=replace(model, host_bytes=host_bytes))
    assert result.target_tokens == 8192
    assert result.requests == 8


@pytest.mark.parametrize(
    "overrides",
    [dict(available_bytes=1), dict(token_limit=63), dict(budget=budget(host_bytes=1))],
)
def test_request_priority_fails_only_when_one_request_cannot_fit(overrides):
    with pytest.raises(MemoryError, match="cannot fit one request"):
        solve_request_priority(**overrides)


def test_request_priority_caps_requests_to_explicit_kda_capacity():
    result = solve_request_priority(explicit_state_slots=7)
    assert result.requests == result.state_slots == 7


def test_request_priority_charges_extra_kda_slots_and_speculative_storage():
    result = solve_request_priority(
        slots_per_request=3,
        explicit_state_slots=32,
        ring_bytes_per_slot=1024,
        intermediate_bytes_per_request=2048,
    )
    assert result.requests == 8
    assert result.state_slots == 32
    assert result.kda_bytes == 33 * ((1 << 20) + 1024) + 9 * 2048


def test_request_priority_keeps_graph_row_floor():
    model = budget(prefill_layer_slots=2, graph_rows=16)
    result = solve_request_priority(budget=model)
    assert model.row_capacity(result.requests) == 16
    assert (
        result.dsa_bytes
        == model.required_bytes(result.target_tokens, result.requests, 128)[1]
    )


@pytest.mark.parametrize("capacity", [1, 3, 7])
def test_request_priority_finds_largest_concurrency_that_fits_hbm(capacity):
    model = budget(prefill_layer_slots=2)
    device = model.required_bytes(capacity * 128, capacity, 128)[1]
    available = device + (capacity + 1) * ((1 << 20) + 32768)
    result = solve_request_priority(available_bytes=available)
    assert result.requests == result.state_slots == capacity
    assert result.target_tokens == capacity * 128
    assert result.unused_bytes == 0


def test_request_priority_reduces_concurrency_for_host_budget():
    model = budget(prefill_layer_slots=2)
    host_bytes = model.required_bytes(3 * 128, 3, 128)[0]
    result = solve_request_priority(budget=replace(model, host_bytes=host_bytes))
    assert result.requests == 3
    assert result.target_tokens == 3 * 128


def test_request_priority_can_run_one_request_under_small_token_cap():
    result = solve_request_priority(token_limit=128)
    assert result.requests == 1
    assert result.target_tokens == 128


def test_request_priority_kda_ratio_limits_effective_requests():
    result = solve_request_priority(slots_per_request=3, explicit_state_slots=7)
    assert result.requests == 2
    assert result.state_slots == 7


def test_request_priority_does_not_exceed_native_query_row_limit():
    model = budget(rows_per_request=4)
    result = solve_request_priority(
        budget=model, available_bytes=1 << 42, requests=10000, token_limit=1 << 30
    )
    assert result.requests == 1024


def test_external_plan_rejects_non_native_page_size():
    with pytest.raises(ValueError, match="geometry"):
        solve_request_priority(page_size=1, token_limit=3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
