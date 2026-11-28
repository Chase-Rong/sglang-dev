"""Optional registered Host KV component adapter for Ascend NPU."""

from dataclasses import dataclass
from typing import Dict

from sglang.srt.hardware_backend.npu.hisparse import host_native


@dataclass
class RegisteredHostComponent:
    """One component region and its lifetime owner."""

    name: str
    region: object
    page_size: int


class NpuRegisteredHostKvPool:
    """Own registered KV components and their device-to-host row copies."""

    def __init__(self, device_id: int = 0, *, shared_group=None):
        self.device_id = device_id
        self.shared_group = shared_group
        self._components: Dict[str, RegisteredHostComponent] = {}
        self._closed = False

    def component(self, name: str) -> RegisteredHostComponent:
        try:
            return self._components[name]
        except KeyError as exc:
            raise KeyError(f"unknown Host KV component {name}") from exc

    def copy_to_host(self, name: str, src, src_index, dst_index, valid_mask):
        result = self.copy_to_host_unpublished(
            name, src, src_index, dst_index, valid_mask
        )
        if self.shared_group is not None:
            self.shared_group.publish_write()
        return result

    def copy_to_host_unpublished(
        self, name: str, src, src_index, dst_index, valid_mask
    ):
        component = self.component(name)
        if self.shared_group is None or self.shared_group.is_writer:
            host_native.copy_rows_to_registered_host(
                src,
                component.region,
                src_index,
                dst_index,
                valid_mask,
            )
        return component.region.tensor

    def close(self, *, drained: bool = False) -> None:
        if self._closed:
            return
        if not drained:
            raise RuntimeError("Host KV users must be drained before close")
        for name in list(self._components):
            self._components[name].region.close(drained=True)
            del self._components[name]
        self._closed = True
