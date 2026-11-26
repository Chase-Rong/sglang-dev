"""Host allocation and prefill capacity for Ascend KV offload."""

from dataclasses import dataclass


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
