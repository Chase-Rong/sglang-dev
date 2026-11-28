"""Admission contracts for the separate Ascend sparse offload lifecycle."""

from contextlib import contextmanager


@contextmanager
def offload_host_backing_budget_scope(plan):
    if plan is None:
        yield
        return
    from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope

    plan.validate()
    with host_memory_budget_scope(plan.scope_available_bytes // plan.shared_ranks):
        yield


def bind_host_memory_budget_plan(backing, plan):
    backing.host_memory_budget_plan = plan
    if plan is not None:
        try:
            backing.host_memory_account.charge(
                "offload_planner", plan.planner_bytes_per_rank
            )
        except BaseException:
            backing.pool.close(drained=True)
            raise


def host_memory_scope_identity():
    from pathlib import Path

    root = Path("/proc")
    membership = tuple(
        sorted(
            line
            for line in (root / "self/cgroup").read_text().splitlines()
            if not line.split(":", 2)[1] or "memory" in line.split(":", 2)[1].split(",")
        )
    )
    mounts = []
    for line in (root / "self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields = after.split()
        if fields[0] == "cgroup2" or (
            fields[0] == "cgroup" and "memory" in fields[2].split(",")
        ):
            mounts.append((fields[0], before.split()[3]))
    namespace = (root / "self/ns/cgroup").stat()
    return (
        (root / "sys/kernel/random/boot_id").read_text().strip(),
        namespace.st_dev,
        namespace.st_ino,
        membership,
        tuple(sorted(mounts)),
    )


def prepare_host_memory_budget_plan(reserve_bytes, *, alignment_bytes=1 << 30):
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
        HostMemoryBudgetPlan,
    )
    from sglang.srt.mem_cache.pool_host.base import (
        host_memory_budget_bytes,
        ranks_per_host,
    )

    group = None
    if envs.SGLANG_NPU_SHARED_HOST_KV.get():
        from sglang.srt.distributed.parallel_state import get_attn_tp_group

        group = get_attn_tp_group()
    shared_ranks = group.world_size if group is not None else 1
    try:
        local_ranks = ranks_per_host()
        if shared_ranks > local_ranks or local_ranks % shared_ranks:
            raise ValueError("Shared Host ranks must partition the local launch ranks")
        identity = host_memory_scope_identity() if shared_ranks > 1 else ()
        available = max(0, host_memory_budget_bytes(auto_size=True))
        report = (
            (identity, reserve_bytes, alignment_bytes, local_ranks),
            available,
            None,
        )
    except Exception as error:
        report = ((), 0, str(error))
    reports = group.all_gather_object(report) if shared_ranks > 1 else [report]
    if any(report[2] is not None for report in reports):
        raise RuntimeError(f"Host memory admission failed across ranks: {reports}")
    if any(report[0] != reports[0][0] for report in reports):
        raise ValueError(
            "Shared Host ranks disagree on the host/cgroup scope, domain reserve or rank placement"
        )
    plan = HostMemoryBudgetPlan(
        reserve_bytes,
        min(report[1] for report in reports) * shared_ranks,
        shared_ranks=shared_ranks,
        scope_identity=reports[0][0][0],
        reserve_alignment_bytes=alignment_bytes,
    )
    plan.validate()
    return plan


def _pool_host_memory_plan(
    kvc, reserve_bytes, *, alignment_bytes=1 << 30, required_bytes=None
):
    if required_bytes is not None and required_bytes > reserve_bytes:
        raise MemoryError("Prefill Host payload exceeds its configured limit")
    plan = getattr(kvc, "_offload_host_memory_plan", None)
    if plan is None:
        plan = getattr(
            getattr(kvc, "memory_pool_config", None), "offload_host_memory_plan", None
        )
    if plan is None and getattr(kvc, "is_draft_worker", False):
        allocator = kvc.token_to_kv_pool_allocator
        pool = allocator.get_kvcache()
        pool = getattr(pool, "full_kv_pool", pool)
        plan = getattr(pool, "host_memory_budget_plan", None)
        if plan is None:
            raise RuntimeError(
                "Draft Host allocation requires the target's Host memory admission"
            )
    if plan is None:
        claim = (
            required_bytes
            if alignment_bytes == 1 and required_bytes is not None
            else reserve_bytes
        )
        plan = prepare_host_memory_budget_plan(claim, alignment_bytes=alignment_bytes)
    if (
        plan.domain_reserve_bytes > reserve_bytes
        or plan.reserve_alignment_bytes != alignment_bytes
        or (alignment_bytes > 1 and plan.domain_reserve_bytes != reserve_bytes)
    ):
        raise ValueError("Target/draft Host memory reservations must match")
    kvc._offload_host_memory_plan = plan
    return plan


def fsa_pool_options(kvc, layer_count):
    import torch

    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import (
        attention_backends,
        get_disagg,
        get_memory,
        get_parallel,
        get_schedule,
    )

    prefill = envs.SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD.get()
    host_budget = envs.SGLANG_NPU_DECODE_HOST_KV_BYTES.get()
    integrated = envs.SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD.get()
    if not (prefill or host_budget or integrated):
        return {}
    if get_disagg().disaggregation_mode != "null":
        raise ValueError("NPU HiSparse requires --disaggregation-mode null")
    if not (
        integrated
        and prefill
        and host_budget > 0
        and envs.SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION.get()
    ):
        raise ValueError(
            "Integrated offload requires Prefill slots, Host budget and fused attention"
        )
    if str(kvc.device).split(":")[0] != "npu" or not kvc.use_mla_backend:
        raise ValueError("FSA Host offload requires an NPU MLA pool")
    if kvc.kv_cache_dtype != torch.bfloat16 or kvc.model_dtype != torch.bfloat16:
        raise ValueError("external_plan requires BF16 query and KV storage")
    parallel = get_parallel()
    if (
        get_schedule().enable_mixed_chunk
        or parallel.attn_cp_size != 1
        or parallel.attn_dcp_size != 1
    ):
        raise ValueError(
            "Integrated offload requires separate Prefill batches and CP1/DCP1"
        )
    if tuple(attention_backends()) != ("ascend", "ascend"):
        raise ValueError("NPU HiSparse requires Ascend for both attention backends")
    memory = get_memory()
    if memory.disable_radix_cache or not memory.enable_hierarchical_cache:
        raise ValueError("Integrated offload requires HiCache and enabled radix cache")
    slots = min(envs.SGLANG_NPU_PREFILL_LAYER_SLOTS.get(), layer_count)
    if slots <= 0:
        raise ValueError("Integrated offload requires positive Prefill layer slots")

    from sglang.srt.hardware_backend.npu.hisparse.host_native import host_ops
    from sglang.srt.hardware_backend.npu.shared_host_kv import (
        shared_host_group_from_runtime,
    )

    host_ops()
    return {
        "prefill_layer_slots": slots,
        "decode_host_budget": host_budget,
        "decode_host_token_capacity": envs.SGLANG_NPU_HICACHE_HOST_TOKENS.get(),
        "shared_host_group": shared_host_group_from_runtime(torch.npu.current_device()),
        "integrated_offload": True,
        "host_memory_budget_plan": _pool_host_memory_plan(kvc, host_budget),
    }
