"""Content-validated immutable topology buffers owned by a draft workspace."""

from collections.abc import Mapping

import torch


def supports_draft_topology(config):
    def value(name, default=None):
        return (
            config.get(name, default)
            if isinstance(config, Mapping)
            else getattr(config, name, default)
        )

    return value("model_type") in ("glm5_next", "glm5_next_text") and bool(
        value("index_share_for_mtp_iteration", False)
    )


def native_topology_ops(device):
    if torch.device(device).type != "npu":
        return None
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops

    return native_ops()


def topology_storage_bytes(batch, rows, topk, width):
    if min(batch, rows, topk, width) <= 0 or batch > rows or width < topk:
        raise ValueError("invalid draft topology reservation")
    return 12 * rows * topk + 12 * batch * width + 92 * batch


class ExternalPlanTopologyCache:
    def __init__(self, device, rows, topk, batch, width, namespace):
        topology_storage_bytes(batch, rows, topk, width)
        if namespace <= 0:
            raise ValueError("draft topology requires a positive model namespace")
        self.namespace = int(namespace)
        self.tensors = (
            torch.full((rows, topk), -1, dtype=torch.int64, device=device),
            torch.full((rows, topk), -1, dtype=torch.int32, device=device),
            torch.full((batch, width), -1, dtype=torch.int64, device=device),
            torch.full((batch, width), -1, dtype=torch.int32, device=device),
            torch.zeros(batch, dtype=torch.int32, device=device),
            torch.zeros((batch, 9), dtype=torch.int64, device=device),
            torch.zeros((batch, 2), dtype=torch.int64, device=device),
        )

    @property
    def statistics(self):
        return self.tensors[-1]

    def invalidate(self):
        self.tensors[-2][:, 0].zero_()
