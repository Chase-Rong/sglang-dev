"""Node-local replicated MLA backing with collective attach and one writer."""

import os
from pathlib import Path

import torch


def local_shared_identity(writer_rank):
    namespace = os.stat("/proc/self/ns/ipc")
    return (
        Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        namespace.st_dev,
        namespace.st_ino,
        writer_rank,
    )


class SharedHostGroup:
    def __init__(self, group, device_id, *, writer_rank=0):
        if not 0 <= writer_rank < group.world_size:
            raise ValueError("shared Host writer rank is outside the TP group")
        self.group = group
        self.device_id = device_id
        self.writer_rank = writer_rank
        self.failed = False
        identity = local_shared_identity(writer_rank)
        peers = group.all_gather_object(identity)
        if any(peer != identity for peer in peers):
            raise ValueError("shared Host TP ranks must share a host and IPC namespace")
        self.fence = torch.zeros(1, dtype=torch.int32, device=f"npu:{device_id}")

    @property
    def is_writer(self):
        return self.group.rank_in_group == self.writer_rank

    def validate_identity(self, identity):
        if self.failed:
            raise RuntimeError("shared Host TP group has failed")
        peers = self.group.all_gather_object(identity)
        if any(peer != identity for peer in peers):
            self.failed = True
            raise ValueError(
                "shared Host ranks disagree on KV geometry or page identity"
            )

    def publish_write(self):
        """Order the owner's preceding copy before every rank's next reader.

        This collective is captured along with current-token writes. It is a
        stream dependency; mapped Host visibility still needs the native gate.
        """
        if self.failed:
            raise RuntimeError("shared Host TP group has failed")
        self.group.broadcast(self.fence, src=self.writer_rank)


def shared_host_group_from_runtime(device_id):
    from sglang.srt.environ import envs

    if not envs.SGLANG_NPU_SHARED_HOST_KV.get():
        return None
    from sglang.srt.distributed.parallel_state import get_attn_tp_group
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.attn_dcp_size != 1:
        raise ValueError("shared MLA Host backing requires replicated KV with DCP=1")
    return SharedHostGroup(get_attn_tp_group(), device_id)
