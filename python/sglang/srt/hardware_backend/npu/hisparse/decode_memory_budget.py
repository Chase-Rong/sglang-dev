"""Capacity accounting for registered Decode KV and persistent FSA state."""

from dataclasses import dataclass
from math import isfinite


def shared_storage_bytes(
    qcap,
    bcap,
    tcap,
    layers,
    topk,
    capacity,
    page_size,
    routes,
    *,
    native_workspace_bytes=0,
    load_mode="FUSED",
    warmup_threshold=0,
    topology_cache_layers=0,
    three_stage=False,
):
    """Exact persistent tensor element bytes for one shared FSA owner.

    CPU generation snapshot is separate; caller's request pool is referenced,
    not cloned here. Graph retained snapshots and tensor-op/native/allocator
    transient peaks are deliberately outside this equality.
    native_workspace_bytes is the complete shared allocation per physical layer.
    """
    if (
        min(qcap, bcap, tcap, layers, topk, capacity, routes) <= 0
        or page_size != 128
        or qcap > 32767
        or bcap > 1024
        or bcap > tcap
        or tcap > 6144
        or capacity > 16384
        or capacity % page_size
        or capacity < routes * topk
        or topk > 2051
        or native_workspace_bytes < 0
        or not 0 <= topology_cache_layers <= layers
    ):
        raise ValueError("invalid external_plan persistent geometry")
    from sglang.srt.hardware_backend.npu.hisparse.config import shared_union_width

    width = shared_union_width(routes, topk)
    if load_mode != "FUSED" or warmup_threshold < 0 or routes > 6:
        raise ValueError("external_plan only supports FUSED and 1..6 routes")
    stride = (topk + 8) // 8 * 8
    directory_width = 1 << (2 * capacity - 1).bit_length()
    per_layer = (
        qcap * capacity * 512 * 2
        + native_workspace_bytes
        + qcap * (11 * capacity + 12 + 4 * directory_width + 64)
        + 11
        + bcap * width * 22
        + bcap * capacity * 8
        + bcap * 145
        + tcap * topk * 20
        + tcap * 12
        + tcap * (32800 + 4 * (capacity // page_size) + 4 * stride)
    )
    if warmup_threshold:
        # Dedicated B_w=1 prefix arrays, no replacement-policy workspace.
        per_layer += 32 * capacity + 96 + 96 * qcap
    if three_stage:
        per_layer += 13 * tcap * topk + 16 * bcap
    topology = topology_cache_layers * (
        12 * tcap * topk + 12 * bcap * width + 92 * bcap
    )
    return {
        "cpu_generation_snapshot": 8 * qcap + (32 * qcap if warmup_threshold else 0),
        "device_owner_global": 38 * qcap + 12,
        "device_layers": layers * per_layer + topology,
    }


def shared_capture_storage_bytes(
    qcap,
    batch,
    rows,
    topk,
    *,
    load_mode="FUSED",
    union_width=None,
    cacheable_mask=False,
    table_columns=0,
    builtin_metadata=False,
    source_table_copy=False,
    three_stage=False,
):
    """Retained output/snapshot storages of one captured layer invocation.

    Allocator-owned intermediate kernel workspaces/capture peak are separate.
    The retained final success mask adds one byte per request-capacity entry.
    Inputs supplied by the model, its query and Attention output are not FSA
    persistent storage. Counts apply to an actually captured bucket/step.
    Builtin sources include conservative TopK normalization and optional device
    table copies; model-owned input storages are charged by the model budget.
    """
    if table_columns < 0:
        raise ValueError("captured table column count must be nonnegative")
    table_bytes = 4 * batch * table_columns * (1 + int(source_table_copy))
    if union_width is not None:
        if not 0 < union_width <= 16384:
            raise ValueError("union lease width must be within the native limit")
        if three_stage:
            return (
                35 * qcap
                + 12 * rows * topk
                + 4 * rows
                + 17 * batch * union_width
                + 92 * batch
                + table_bytes
            )
        return (
            35 * qcap
            + (20 + int(cacheable_mask or builtin_metadata)) * rows * topk
            + (4 * rows * topk if builtin_metadata else 0)
            + 4 * rows
            + 17 * batch * union_width
            + 92 * batch
            + 8
            + table_bytes
        )
    if three_stage:
        raise ValueError("three-stage capture accounting requires native union width")
    return 35 * qcap + 34 * rows * topk + 24 * rows + 80 * batch + 8 + table_bytes


@dataclass(frozen=True)
class HostMemoryBudgetPlan:
    domain_reserve_bytes: int
    scope_available_bytes: int
    shared_ranks: int = 1
    planner_bytes_per_rank: int = 0
    scope_identity: tuple = ()
    reserve_alignment_bytes: int = 1 << 30

    def __post_init__(self):
        if (
            min(
                self.domain_reserve_bytes,
                self.shared_ranks,
                self.reserve_alignment_bytes,
            )
            <= 0
        ):
            raise ValueError(
                "Host domain reserve and shared rank count must be positive"
            )
        if min(self.scope_available_bytes, self.planner_bytes_per_rank) < 0:
            raise ValueError("Host scope and planner budgets must be nonnegative")

    @property
    def aligned_reserve_bytes(self):
        alignment = self.reserve_alignment_bytes
        return (self.domain_reserve_bytes + alignment - 1) // alignment * alignment

    def required_scope_bytes(self, private_bytes_per_rank=0):
        if private_bytes_per_rank < 0:
            raise ValueError("Private Host storage must be nonnegative")
        return self.aligned_reserve_bytes + self.shared_ranks * (
            self.planner_bytes_per_rank + private_bytes_per_rank
        )

    @property
    def sidecar_budget_bytes(self):
        return max(
            0,
            (self.scope_available_bytes - self.aligned_reserve_bytes)
            // self.shared_ranks
            - self.planner_bytes_per_rank,
        )

    def physical_charge_bytes(self, *, is_writer):
        return self.aligned_reserve_bytes if is_writer else 0

    def validate(self, private_bytes_per_rank=0):
        required = self.required_scope_bytes(private_bytes_per_rank)
        if required > self.scope_available_bytes:
            raise MemoryError(
                f"Host scope requires {required} bytes for one aligned MemFabric "
                f"domain and {self.shared_ranks} private planner/sidecar sets; "
                f"available after cgroup/OS reserve is {self.scope_available_bytes}"
            )


@dataclass(frozen=True)
class PrefillKvMemoryBudget:
    host_bytes_per_token: int
    device_bytes_per_token: int
    host_bytes: int
    device_fixed_bytes: int = 0
    host_token_capacity: int = 0

    def __post_init__(self):
        if (
            min(self.host_bytes_per_token, self.device_bytes_per_token, self.host_bytes)
            <= 0
            or self.device_fixed_bytes < 0
        ):
            raise ValueError(
                "Prefill offload requires positive storage and nonnegative fixed bytes"
            )

        if type(self.host_token_capacity) is not int or self.host_token_capacity < 0:
            raise ValueError("Prefill Host token capacity must be nonnegative")

    def max_tokens(self, device_bytes, page_size):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.host_token_capacity % page_size:
            raise ValueError("Prefill Host capacity must be page aligned")
        if (
            self.host_token_capacity + page_size
        ) * self.host_bytes_per_token > self.host_bytes:
            return 0
        rows = min(
            max(0, device_bytes - self.device_fixed_bytes)
            // self.device_bytes_per_token,
            self.host_bytes // self.host_bytes_per_token,
        )
        return max(0, (rows // page_size - 1) * page_size)

    def validate_host_capacity(self, token_count, page_size):
        if token_count < 0 or page_size <= 0 or token_count % page_size:
            raise ValueError("invalid Prefill capacity")
        if self.host_token_capacity % page_size:
            raise ValueError("Prefill Host capacity must be page aligned")
        if (
            max(token_count, self.host_token_capacity) + page_size
        ) * self.host_bytes_per_token > self.host_bytes:
            raise MemoryError("Prefill Host KV exceeds its byte budget")


@dataclass(frozen=True)
class DecodeKvMemoryBudget:
    """Candidate capacity budget with a baseline native_workspace_bytes value.

    The default describes the AIC32 development platform; startup queries the
    actual platform workspace instead. There is no standalone finish reserve.
    """

    layers: int
    latent_dim: int
    rope_dim: int
    index_dim: int
    element_bytes: int
    topk: int
    resident_capacity: int
    host_bytes: int
    graph_rows: int = 0
    callback_instances: int = 1
    shared_fence_bytes: int = 0
    staging_bytes: int = 0
    rows_per_request: int = 1
    prefill_layer_slots: int = 0
    host_token_capacity: int = 0
    index_layers: int | None = None
    index_scale_bytes: int = 0
    index_tail_bytes_per_request: int = 0
    device_fixed_bytes: int = 0
    external_request_capacity: int = 0
    external_request_extra_slots: int = 0
    external_resident_instances: int = 1
    host_memory_plan: HostMemoryBudgetPlan | None = None
    host_index_kpool: int = 1
    host_index_sidecar: bool = False
    execution_abi: str | None = None
    shared_graph_executions: tuple = ()
    shared_table_columns: int = 0
    # A5 Ascend950DT_9582, AIC32/libapi16MiB. Production factory replaces this
    # with the queried platform value before solving capacity.
    native_workspace_bytes: int = 30932992
    load_mode: str | None = None
    warmup_threshold: int | None = None
    topology_cache_layers: int = 0
    three_stage: bool = False

    def __post_init__(self):
        from .config import (
            resolve_fsa_execution_abi,
            resolve_shared_load_mode,
            resolve_shared_warmup_threshold,
        )

        object.__setattr__(
            self, "execution_abi", resolve_fsa_execution_abi(self.execution_abi)
        )
        object.__setattr__(self, "load_mode", resolve_shared_load_mode(self.load_mode))
        object.__setattr__(
            self,
            "warmup_threshold",
            resolve_shared_warmup_threshold(self.warmup_threshold),
        )
        if self.latent_dim != 512 or self.rope_dim != 0 or self.element_bytes != 2:
            raise ValueError("external_plan budget requires BF16/NoPE512/RoPE0")
        if self.resident_capacity > 16384 or self.rows_per_request > 6:
            raise ValueError("FUSED requires C<=16384 and routes<=6")
        if self.native_workspace_bytes < 0:
            raise ValueError("native workspace budget must be nonnegative")
        if not 0 <= self.topology_cache_layers <= self.layers:
            raise ValueError(
                "topology cache layers must fit the logical KV layer count"
            )
        if self.host_index_kpool <= 0:
            raise ValueError("Host Indexer compression must be positive")
        if (
            self.host_memory_plan is not None
            and self.host_memory_plan.domain_reserve_bytes != self.host_bytes
        ):
            raise ValueError("Decode payload and Host domain budgets must match")
        if self.index_layers is not None and not 0 <= self.index_layers <= self.layers:
            raise ValueError("Index layers must fit the logical KV layer count")
        if (
            min(
                self.index_scale_bytes,
                self.index_tail_bytes_per_request,
                self.device_fixed_bytes,
                self.external_request_capacity,
                self.external_request_extra_slots,
                self.shared_table_columns,
            )
            < 0
        ):
            raise ValueError("Index and fixed allocation bytes must be nonnegative")
        if self.external_resident_instances <= 0:
            raise ValueError("external resident instance count must be positive")
        if (
            min(
                self.layers,
                self.latent_dim,
                self.index_dim,
                self.element_bytes,
                self.host_bytes,
                self.callback_instances,
            )
            <= 0
        ):
            raise ValueError("Decode offload requires positive dimensions and budgets")
        if self.rope_dim < 0:
            raise ValueError("RoPE dimension must be nonnegative")
        if (
            not 0 < self.topk <= 2051
            or not self.topk <= self.resident_capacity <= 32767
        ):
            raise ValueError("Decode selection capacity is outside the fused plan ABI")
        if self.graph_rows < 0 or self.shared_fence_bytes < 0:
            raise ValueError("invalid graph capacity or shared fence size")
        if self.staging_bytes < 0 or self.staging_bytes % 32:
            raise ValueError(
                "Decode staging budget must be nonnegative and 32-byte aligned"
            )
        if self.rows_per_request <= 0:
            raise ValueError("Decode query rows per request must be positive")
        if not 0 <= self.prefill_layer_slots <= self.layers:
            raise ValueError("Prefill slots must fit the Host layer count")
        if type(self.host_token_capacity) is not int or self.host_token_capacity < 0:
            raise ValueError("Host token capacity must be a nonnegative integer")

    @property
    def host_bytes_per_token(self):
        return self.layers * (self.latent_dim + self.rope_dim) * self.element_bytes

    @property
    def device_bytes_per_token(self):
        index_layers = self.layers if self.index_layers is None else self.index_layers
        return (
            index_layers
            * (self.index_dim * self.element_bytes + self.index_scale_bytes)
            // self.host_index_kpool
            + self.prefill_layer_slots
            * (self.latent_dim + self.rope_dim)
            * self.element_bytes
        )

    def allocation_overhead(self, requests):
        return (
            self.external_request_slots(requests) * self.index_tail_bytes_per_request
            + self.device_fixed_bytes
        )

    def selection_capacity(self, page_size):
        if page_size <= 0:
            raise ValueError("page size must be positive")
        capacity = (self.resident_capacity + page_size - 1) // page_size * page_size
        if capacity > 32767:
            raise ValueError(
                "page-aligned selection capacity exceeds the fused plan ABI"
            )
        return capacity

    def row_capacity(self, requests):
        return max(requests, self.graph_rows) * self.rows_per_request

    def external_request_slots(self, requests):
        """Price the candidate pool before it exists, including reserved rows."""
        return max(
            requests + self.external_request_extra_slots + 1,
            self.external_request_capacity,
        )

    def shared_persistent_bytes(self, request_rows, page_size):
        qcap = self.external_request_slots(request_rows)
        bcap = max(request_rows, self.graph_rows)
        tcap = self.row_capacity(request_rows)
        values = shared_storage_bytes(
            qcap,
            bcap,
            tcap,
            1,
            self.topk,
            self.selection_capacity(page_size),
            page_size,
            self.rows_per_request,
            native_workspace_bytes=self.native_workspace_bytes,
            load_mode=self.load_mode,
            warmup_threshold=self.warmup_threshold,
            three_stage=self.three_stage,
        )
        topology = 0
        if self.topology_cache_layers:
            from sglang.srt.hardware_backend.npu.hisparse.config import (
                shared_union_width,
            )
            from sglang.srt.hardware_backend.npu.hisparse.external_plan_topology import (
                topology_storage_bytes,
            )

            topology = self.topology_cache_layers * topology_storage_bytes(
                bcap,
                tcap,
                self.topk,
                shared_union_width(self.rows_per_request, self.topk),
            )
        return {
            "cpu": values["cpu_generation_snapshot"] * self.external_resident_instances,
            "device": values["device_owner_global"] * self.external_resident_instances
            + values["device_layers"] * self.layers
            + topology,
        }

    def fixed_bytes(self, request_rows, page_size):
        # Input capacity is expressed in allocator pages. The native planner
        # sees 128-token attention pages, including on compressed KPool models.
        if page_size != 128 * self.host_index_kpool:
            raise ValueError("invalid external_plan allocator page geometry")
        page_size = 128
        from .config import shared_union_width

        persistent = self.shared_persistent_bytes(request_rows, page_size)
        qcap = self.external_request_slots(request_rows)
        bcap = max(request_rows, self.graph_rows)
        tcap = self.row_capacity(request_rows)
        capacity = self.selection_capacity(page_size)
        width = shared_union_width(self.rows_per_request, self.topk)
        # Explicit bucket/step shapes give exact retained FSA snapshot bytes.
        # If capture shapes are not available yet, callback_instances reserves
        # an upper bound; this is not reported as measured persistent equality.
        captures = self.shared_graph_executions
        if not captures and self.graph_rows:
            captures = ((bcap, tcap),) * self.callback_instances
        snapshots = self.layers * sum(
            shared_capture_storage_bytes(
                qcap,
                b,
                t,
                self.topk,
                load_mode=self.load_mode,
                union_width=width,
                cacheable_mask=True,
                table_columns=self.shared_table_columns,
                builtin_metadata=not self.warmup_threshold,
                source_table_copy=not self.warmup_threshold,
                three_stage=self.three_stage,
            )
            for b, t in captures
        )
        # Tensor-operation scratch allowance, not a measurement of native ACL
        # workspace, allocator/capture peak, or a persistent allocation.
        temporary = self.layers * (
            bcap * (width * 160 + capacity * 96) + tcap * self.topk * 96
        )
        if self.warmup_threshold:
            temporary += self.layers * capacity * (160 + 96 + 96 + 32)
        # Native workspace is one persistent buffer per serialized physical
        # layer, already included above. Captured bucket/step aliases add zero.
        return (
            persistent["cpu"]
            + (4 * self.layers * len(captures) if self.three_stage else 0),
            persistent["device"]
            + snapshots
            + temporary
            + self.shared_fence_bytes
            + self.staging_bytes
            + self.allocation_overhead(request_rows),
        )

    def required_bytes(self, tokens, request_rows, page_size):
        if page_size <= 0 or tokens < 0 or tokens % page_size:
            raise ValueError(
                "Decode token capacity must be nonnegative and page aligned"
            )
        host_fixed, device_fixed = self.fixed_bytes(request_rows, page_size)
        if self.host_token_capacity % page_size:
            raise ValueError("Host token capacity must be page aligned")
        storage_rows = tokens + page_size
        host_storage_rows = max(tokens, self.host_token_capacity) + page_size
        return (
            (host_fixed if self.host_memory_plan is None else 0)
            + host_storage_rows * self.host_bytes_per_token,
            device_fixed + storage_rows * self.device_bytes_per_token,
        )

    def host_indexer_bytes(self, tokens, page_size):
        if not self.host_index_sidecar:
            return 0
        if page_size <= 0 or page_size % self.host_index_kpool:
            raise ValueError(
                "Host Indexer pages must contain complete compression groups"
            )
        layers = self.layers if self.index_layers is None else self.index_layers
        slots = page_size // self.host_index_kpool
        key_bytes = (slots * self.index_dim * self.element_bytes + 7) // 8 * 8
        scale_bytes = (slots * self.index_scale_bytes + 7) // 8 * 8
        pages = max(tokens, self.host_token_capacity) // page_size + 1
        return pages * layers * (key_bytes + scale_bytes)

    def host_scope_fits(self, tokens, requests, page_size, extra_host_bytes=0):
        if self.host_memory_plan is None:
            return True
        planner, _ = self.fixed_bytes(requests, page_size)
        private = (
            max(0, planner - self.host_memory_plan.planner_bytes_per_rank)
            + self.host_indexer_bytes(tokens, page_size)
            + extra_host_bytes
        )
        return (
            self.host_memory_plan.required_scope_bytes(private)
            <= self.host_memory_plan.scope_available_bytes
        )

    def max_tokens(
        self, device_bytes, page_size, request_rows_for_tokens, *, extra_host_bytes=0
    ):
        if page_size <= 0 or device_bytes < 0:
            raise ValueError("invalid Decode page size or NPU byte budget")
        lower = 0
        upper = (
            min(
                self.host_bytes // self.host_bytes_per_token,
                (
                    device_bytes // self.device_bytes_per_token
                    if self.device_bytes_per_token
                    else self.host_bytes // self.host_bytes_per_token
                ),
            )
            // page_size
        )
        while lower < upper:
            candidate = (lower + upper + 1) // 2
            tokens = candidate * page_size
            rows = max(1, request_rows_for_tokens(tokens))
            host_required, device_required = self.required_bytes(
                tokens, rows, page_size
            )
            if (
                host_required <= self.host_bytes
                and device_required <= device_bytes
                and self.host_scope_fits(tokens, rows, page_size, extra_host_bytes)
            ):
                lower = candidate
            else:
                upper = candidate - 1
        return lower * page_size

    def validate(self, tokens, request_rows, page_size, device_bytes=None):
        host_required, device_required = self.required_bytes(
            tokens, request_rows, page_size
        )
        if host_required > self.host_bytes:
            raise MemoryError(
                f"Decode Host backing and planner require {host_required} bytes; budget {self.host_bytes}"
            )
        if not self.host_scope_fits(tokens, request_rows, page_size):
            raise MemoryError(
                "Decode Host scope cannot fit the aligned domain and private sidecars"
            )
        if device_bytes is not None and device_required > device_bytes:
            raise MemoryError(
                f"Decode Indexer and selection require {device_required} bytes; budget {device_bytes}"
            )


@dataclass(frozen=True)
class JointDsaKdaCapacity:
    target_tokens: int
    requests: int
    state_slots: int
    dsa_bytes: int
    kda_bytes: int
    unused_bytes: int
    request_bytes: int = 0


def solve_joint_dsa_kda_tokens(
    budget,
    available_bytes,
    token_limit,
    page_size,
    requests,
    state_bytes_per_slot,
    slots_per_request=1,
    ring_bytes_per_slot=0,
    intermediate_bytes_per_request=0,
    explicit_state_slots=None,
    request_bytes_per_row=0,
    fixed_state_bytes=0,
    host_state_bytes_per_slot=0,
    host_to_device_ratio=1,
):
    """Prefer the requested concurrency, then maximize the shared token pool."""
    if min(token_limit, page_size, requests, slots_per_request) <= 0:
        raise ValueError("joint capacity dimensions must be positive")
    if (
        min(
            available_bytes,
            state_bytes_per_slot,
            fixed_state_bytes,
            ring_bytes_per_slot,
            intermediate_bytes_per_request,
            request_bytes_per_row,
            host_state_bytes_per_slot,
        )
        < 0
    ):
        raise ValueError("joint capacity byte budgets must be nonnegative")
    if explicit_state_slots is not None and explicit_state_slots <= 0:
        raise ValueError("explicit KDA state slots must be positive")
    if not isfinite(host_to_device_ratio) or host_to_device_ratio <= 0:
        raise ValueError("Host checkpoint ratio must be finite and positive")
    budget.fixed_bytes(1, page_size)
    minimum_per_request = max(page_size, 2)
    upper = min(
        requests,
        token_limit // minimum_per_request,
        1024,
        6144 // budget.rows_per_request,
    )
    if explicit_state_slots is not None:
        upper = min(upper, explicit_state_slots // slots_per_request)

    def fits(candidate_requests):
        slots = (
            candidate_requests * slots_per_request
            if explicit_state_slots is None
            else explicit_state_slots
        )
        minimum_tokens = (
            (candidate_requests * minimum_per_request + page_size - 1)
            // page_size
            * page_size
        )
        host_bytes, device_bytes = budget.required_bytes(
            minimum_tokens, candidate_requests, page_size
        )
        state_bytes = (
            (slots + 1) * (state_bytes_per_slot + ring_bytes_per_slot)
            + (candidate_requests + 1) * intermediate_bytes_per_request
            + fixed_state_bytes
        )
        mapping_bytes = (candidate_requests + 1) * request_bytes_per_row
        return (
            host_bytes <= budget.host_bytes
            and device_bytes + state_bytes + mapping_bytes <= available_bytes
            and budget.host_scope_fits(
                minimum_tokens,
                candidate_requests,
                page_size,
                (int(slots * host_to_device_ratio) + 1) * host_state_bytes_per_slot,
            )
        )

    lower = 0
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        if fits(midpoint):
            lower = midpoint
        else:
            upper = midpoint - 1
    if lower == 0:
        raise MemoryError(
            "Joint DSA/KDA budget cannot fit one request with the configured "
            f"state, Graph and token limits: requested={requests}, "
            f"available_bytes={available_bytes}"
        )
    requests = lower
    state_slots = (
        requests * slots_per_request
        if explicit_state_slots is None
        else explicit_state_slots
    )
    kda_bytes = (
        (state_slots + 1) * (state_bytes_per_slot + ring_bytes_per_slot)
        + (requests + 1) * intermediate_bytes_per_request
        + fixed_state_bytes
    )
    request_bytes = (requests + 1) * request_bytes_per_row
    dsa_available = available_bytes - kda_bytes - request_bytes
    tokens = min(
        token_limit // page_size * page_size,
        budget.max_tokens(
            dsa_available,
            page_size,
            lambda _: requests,
            extra_host_bytes=(int(state_slots * host_to_device_ratio) + 1)
            * host_state_bytes_per_slot,
        ),
    )
    _, dsa_bytes = budget.required_bytes(tokens, requests, page_size)
    return JointDsaKdaCapacity(
        tokens,
        requests,
        state_slots,
        dsa_bytes,
        kda_bytes,
        available_bytes - dsa_bytes - kda_bytes - request_bytes,
        request_bytes,
    )


def budget_from_config(kvc, layers):
    import torch

    from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.hisparse.config import (
        resolve_fsa_execution_abi,
    )
    from sglang.srt.hardware_backend.npu.hisparse.offload_config import (
        layer_staging_bytes,
    )
    from sglang.srt.hardware_backend.npu.hisparse.speculative_sparse import (
        sparse_mtp_draft_layers,
        sparse_mtp_rows,
    )
    from sglang.srt.runtime_context import get_disagg, get_exec, get_spec

    if (
        str(kvc.device).split(":")[0] != "npu"
        or not kvc.use_mla_backend
        or not is_deepseek_dsa(kvc.model_config.hf_config)
        or get_disagg().disaggregation_mode
        != ("null" if envs.SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD.get() else "decode")
    ):
        raise ValueError("registered Decode sizing requires an NPU DSA Decode service")
    if kvc.kv_cache_dtype != torch.bfloat16 or kvc.model_dtype != torch.bfloat16:
        raise ValueError("external_plan requires BF16 query and KV storage")
    model = kvc.model_config
    from sglang.srt.model_executor.pool_configurator import _get_dsa_cache_layer_ids

    layer_ids = _get_dsa_cache_layer_ids(kvc, layers)
    draft_layers = sparse_mtp_draft_layers(kvc)
    from sglang.srt.mem_cache.kv_cache_configurator import _should_elide_dsa_index_k

    active_index_layers = sum(
        not dsa_layer_skips_topk(model.hf_config, layer) for layer in layer_ids
    )
    index_layers = (
        active_index_layers
        if kvc.mambaish_config is None
        and _should_elide_dsa_index_k(is_draft_worker=kvc.is_draft_worker)
        else len(layer_ids)
    ) + draft_layers
    index_kpool = int(getattr(model.hf_text_config, "index_kpool", 1))
    from sglang.srt.configs.model_config import get_dsa_index_kpool_compress
    from sglang.srt.runtime_context import max_speculative_num_draft_tokens

    tail_bytes = 0
    if index_kpool > 1 and get_dsa_index_kpool_compress(model.hf_config):
        tail_bytes = (
            (active_index_layers + draft_layers)
            * (index_kpool + (max_speculative_num_draft_tokens() or 0))
            * model.index_head_dim
            * 2
            * 2
        )
    prefill_slots = 0
    if envs.SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD.get():
        configured_slots = envs.SGLANG_NPU_PREFILL_LAYER_SLOTS.get()
        prefill_slots = min(configured_slots, layers) + min(
            configured_slots, draft_layers
        )
    rows_per_request = sparse_mtp_rows(kvc.spec_algorithm, get_spec(), model)
    expanded_topk = int(model.hf_text_config.index_topk) + index_kpool - 1
    configured_capacity = envs.SGLANG_NPU_FSA_RESIDENT_CAPACITY.get()
    from .config import validate_dsa_offload_backend
    from .external_sparse_planner import expanded_resident_capacity

    validate_dsa_offload_backend(envs.SGLANG_NPU_DSA_OFFLOAD_BACKEND.get())
    resident_capacity = expanded_resident_capacity(
        configured_capacity, rows_per_request, expanded_topk, 128
    )
    layers += draft_layers
    external_request_capacity = 0
    external_request_extra_slots = 0
    pool = getattr(kvc, "req_to_token_pool", None)
    generation = getattr(pool, "req_generation", None)
    if generation is not None:
        external_request_capacity = int(generation.numel())
    disagg = get_disagg()
    if disagg.disaggregation_mode == "decode" or getattr(
        disagg, "enable_pd_role_switch", False
    ):
        external_request_extra_slots = (
            getattr(disagg, "disaggregation_decode_extra_slots", 0) or 0
        )
    graph_config = get_exec().graph.cuda_graph_config.decode
    graph_sizes = () if graph_config.backend == "disabled" else (graph_config.bs or ())
    graph_rows = max(graph_sizes, default=0)
    if graph_config.backend != "disabled":
        graph_rows = max(graph_rows, graph_config.max_bs or 0)
    execution_abi = resolve_fsa_execution_abi()
    topology_cache_layers = 0
    from sglang.srt.hardware_backend.npu.hisparse.config import (
        register_shared_execution,
        resolve_shared_load_mode,
        resolve_shared_warmup_threshold,
    )

    load_mode = resolve_shared_load_mode()
    warmup_threshold = resolve_shared_warmup_threshold()
    from sglang.srt.hardware_backend.npu.fsa_native import (
        shared_workspace_bytes,
    )

    native_workspace = shared_workspace_bytes()
    if pool is not None:
        register_shared_execution(pool, load_mode, warmup_threshold)
    from sglang.srt.hardware_backend.npu.hisparse.external_plan_topology import (
        native_topology_ops,
        supports_draft_topology,
    )

    cache_layers = len(layer_ids) if kvc.is_draft_worker else draft_layers
    if cache_layers and supports_draft_topology(model.hf_text_config):
        if native_topology_ops(kvc.device) is not None:
            topology_cache_layers = cache_layers
    return DecodeKvMemoryBudget(
        layers,
        model.kv_lora_rank,
        model.qk_rope_head_dim,
        model.index_head_dim,
        2,
        expanded_topk,
        resident_capacity,
        envs.SGLANG_NPU_DECODE_HOST_KV_BYTES.get(),
        graph_rows,
        max(1, len(graph_sizes) if graph_sizes else graph_rows)
        * (
            1
            if kvc.spec_algorithm.is_none()
            else int(get_spec().speculative_num_steps) + 2
        )
        + 1,
        shared_fence_bytes=4 if envs.SGLANG_NPU_SHARED_HOST_KV.get() else 0,
        staging_bytes=layer_staging_bytes(
            envs.SGLANG_NPU_LAYER_STAGING_BYTES.get(),
            role=get_disagg().disaggregation_mode,
            host_offload=envs.SGLANG_NPU_DECODE_HOST_KV_BYTES.get() > 0,
            integrated=envs.SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD.get(),
        ),
        rows_per_request=rows_per_request,
        prefill_layer_slots=prefill_slots,
        host_token_capacity=(envs.SGLANG_NPU_HICACHE_HOST_TOKENS.get()),
        execution_abi=execution_abi,
        native_workspace_bytes=native_workspace,
        three_stage=True,
        topology_cache_layers=topology_cache_layers,
        shared_graph_executions=tuple(
            (b, b * rows_per_request)
            for b in graph_sizes
            for _ in range(
                1
                if kvc.spec_algorithm.is_none()
                else int(get_spec().speculative_num_steps) + 2
            )
        ),
        shared_table_columns=(model.context_len + 255) // 128,
        index_layers=index_layers,
        index_tail_bytes_per_request=tail_bytes,
        host_index_kpool=index_kpool,
        host_index_sidecar=index_kpool > 1
        and get_dsa_index_kpool_compress(model.hf_config),
        external_request_capacity=external_request_capacity,
        external_request_extra_slots=external_request_extra_slots,
        external_resident_instances=1 + int(draft_layers > 0),
    )
