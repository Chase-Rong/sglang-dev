"""The single external_plan fused planner/SFA execution state."""

import torch


def paged_kv_view(tensor: torch.Tensor, page_size: int, head_dim: int) -> torch.Tensor:
    """Expose split PA storage without copying the full context to HBM."""
    if tensor.dim() == 4 and tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    if (
        tensor.dim() != 3
        or tensor.shape[1:] != (page_size, head_dim)
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "fused KV requires contiguous split [pages,page_size,1,dim] storage"
        )
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("fused KV requires FP16 or BF16 storage")
    return tensor


class FusedSparseAttentionState:
    """Per-layer resident and workspace ownership for the sole external path."""

    def __init__(
        self,
        device,
        req_to_token_pool,
        *,
        resident_capacity=None,
        layer_ids=None,
        row_capacity=None,
        mutable_tail_tokens=0,
        max_query_routes=1,
        page_size=128,
        execution_abi=None,
        batch_capacity=None,
        load_mode=None,
        warmup_threshold=None,
        draft_topology_reuse=False,
    ):
        from .config import (
            resolve_fsa_execution_abi,
            resolve_shared_load_mode,
            resolve_shared_warmup_threshold,
        )
        from .shared_execution import _INSTANCES

        self.execution_abi = resolve_fsa_execution_abi(execution_abi)
        self._load_mode = resolve_shared_load_mode(load_mode)
        self._warmup_threshold = resolve_shared_warmup_threshold(warmup_threshold)
        self.device = torch.device(device)
        if self.device.type == "npu" and self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.pool = req_to_token_pool
        self.shared_instance = next(_INSTANCES)
        self.topology_ops = None
        if draft_topology_reuse:
            from .external_plan_topology import native_topology_ops

            self.topology_ops = native_topology_ops(self.device)
        self.batch_capacity = batch_capacity or min(
            row_capacity or req_to_token_pool.req_generation.numel() - 1,
            max(
                req_to_token_pool.req_generation.numel() - 1,
                ((row_capacity or 1) + max_query_routes - 1) // max_query_routes,
            ),
        )
        if (
            not 0 < self.batch_capacity <= 1024
            or req_to_token_pool.req_generation.numel() > 32767
            or page_size != 128
        ):
            raise ValueError(
                "external_plan requires page128, Bcap<=1024 and Qcap<=32767"
            )
        if not 0 < max_query_routes <= 6 or mutable_tail_tokens < 0:
            raise ValueError(
                "external_plan requires 1..6 routes and a nonnegative mutable tail"
            )
        self.shared_native_workspace_bytes = 0
        self.three_stage_enabled = self.device.type == "npu"
        if self.three_stage_enabled:
            from sglang.srt.hardware_backend.npu.fsa_native import (
                shared_workspace_bytes,
            )

            self.shared_native_workspace_bytes = shared_workspace_bytes()
        self.layers = {}
        self.resident_capacity = resident_capacity
        self.layer_ids = layer_ids
        self.row_capacity = row_capacity
        self.mutable_tail_tokens = mutable_tail_tokens
        self.max_query_routes = max_query_routes
        self.page_size = page_size
        self.external_resident = None
        self.mapping_epoch_input = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )
        self._mapping_epoch_host = None
        self.generations = torch.empty_like(
            self.pool.req_generation, device=self.device
        )
        self._generation_snapshot = self.pool.req_generation.clone()
        self._pool_epoch = getattr(self.pool, "req_generation_epoch", 0)
        self._epoch = 0
        self._generations_initialized = False
        self._generation_version = 0
        self.refresh_generations()

    @property
    def load_mode(self):
        return self._load_mode

    @property
    def warmup_threshold(self):
        return self._warmup_threshold

    def invalidate_warmup_content(self, owners):
        """Explicit same-prefix overwrite/replacement hook; append needs none."""
        for layer in self.layers.values():
            if hasattr(layer, "warmup"):
                layer.guard.assert_stream()
                layer.warmup.invalidate(owners)

    def refresh_mapping_epoch(self, mapping_epoch):
        """Refresh the fixed-address resident invalidation epoch outside the graph."""
        stream = (
            torch.npu.current_stream()
            if self.layers and self.device.type == "npu"
            else None
        )
        for layer in self.layers.values():
            layer.guard.assert_stream(stream)
        if (
            self.warmup_threshold
            and getattr(self, "_warmup_epoch", None) != mapping_epoch
        ):
            for layer in self.layers.values():
                if hasattr(layer, "warmup"):
                    layer.warmup.marker.fill_(-1)
                    layer.warmup.stamp.fill_(-1)
            self._warmup_epoch = mapping_epoch
        if self.three_stage_enabled:
            if self._mapping_epoch_host == mapping_epoch:
                return
            self._mapping_epoch_host = None
            self.mapping_epoch_input.fill_(mapping_epoch)
            self._mapping_epoch_host = mapping_epoch
        else:
            self.mapping_epoch_input.fill_(mapping_epoch)

    def refresh_generations(self):
        """Refresh outside capture; graph replay reads this stable device buffer."""
        stream = (
            torch.npu.current_stream()
            if self.layers and self.device.type == "npu"
            else None
        )
        for layer in self.layers.values():
            layer.guard.assert_stream(stream)
        current = self.pool.req_generation
        pool_epoch = getattr(self.pool, "req_generation_epoch", 0)
        # Request generations live on CPU and usually stay unchanged throughout
        # decode. Avoid a blocking H2D copy while still advancing plan versions.
        if (
            self._generations_initialized
            and pool_epoch == self._pool_epoch
            and torch.equal(current, self._generation_snapshot)
        ):
            return
        # A failed upload must be retried even after the CPU snapshot advances.
        self._generations_initialized = False
        if (
            pool_epoch != self._pool_epoch
            or torch.any(current < self._generation_snapshot).item()
        ):
            self._epoch += 1
        if self.warmup_threshold:
            changed = current != self._generation_snapshot
            if pool_epoch != self._pool_epoch:
                changed = torch.ones_like(changed, dtype=torch.bool)
            for layer in self.layers.values():
                if hasattr(layer, "warmup"):
                    device_changed = changed.to(device=self.device)
                    layer.warmup.marker.masked_fill_(device_changed[:, None], -1)
                    layer.warmup.stamp.masked_fill_(device_changed[:, None], -1)
        self._pool_epoch = pool_epoch
        self._generation_snapshot.copy_(current)
        identities = current + (self._epoch << 32)
        self.generations.copy_(identities)
        self._generation_version += 1
        self._generations_initialized = True

    def layer(self, layer_id):
        if layer_id not in self.layers:
            raise RuntimeError("shared layer must be reserved with its geometry first")
        return self.layers[layer_id]

    def reserve(self, layer_id, token_capacity, topk, full_rope, full_kv):
        from .shared_execution import reserve_shared

        return reserve_shared(self, layer_id, token_capacity, topk, full_rope, full_kv)

    def build_inputs(self, **args):
        from .shared_execution import build_shared_inputs

        args.setdefault("query_positions", None)
        args.setdefault("mapping_epoch", 0)
        return build_shared_inputs(self, **args)


def run_fused_sparse_attention(inputs, mode="fused"):
    from .shared_execution import run_shared

    return run_shared(inputs, mode)
