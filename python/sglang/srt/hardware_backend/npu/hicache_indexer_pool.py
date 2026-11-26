"""HiCache sidecar for compressed NPU DSA indexer pages.

An allocator page contains one complete compressed indexer page. The sidecar
stores that page under the same Host allocation as its latent KV rows.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.hardware_backend.npu.hicache_indexer_mapping import (
    prepare_npu_hicache_indices,
)
from sglang.srt.mem_cache.device_pool_info import MLASplitBuffers
from sglang.srt.mem_cache.pool_host.base import HostKVCache, host_memory_budget_bytes

logger = logging.getLogger(__name__)


def _native_index_buffers(pool):
    (info,) = pool.get_device_pool_infos()
    info.buffer_info.validate(layer_ids=info.layer_ids)
    native = info.buffer_info.buffers
    if not isinstance(native, MLASplitBuffers):
        raise ValueError("NPU Indexer requires explicit native split buffers")
    return native


def npu_indexer_host_bytes_per_layer_page(pool):
    native = _native_index_buffers(pool)
    if native.index_k is None or not native.index_layer_indices:
        raise ValueError("NPU Indexer declaration requires live index layers")
    per_layer = 0
    for buffer in (native.index_k, native.index_scales):
        if buffer is not None:
            per_layer += (
                (
                    pool.page_size
                    // pool.index_kpool
                    * buffer[0, 0, 0].numel()
                    * buffer.element_size()
                    + 7
                )
                // 8
                * 8
            )
    return per_layer


class NPUIndexerPoolHost(HostKVCache):
    """Store compressed slices under the anchor's uncompressed KV page IDs."""

    prepare_transfer_indices = staticmethod(prepare_npu_hicache_indices)

    def __init__(
        self, device_pool, anchor_host, *, pin_memory=True, mtp_draft_device_pools=None
    ):
        self.device_pool = device_pool
        self.anchor_host = anchor_host
        self.binding = anchor_host.binding
        self.shared_allocation_domain = None
        self.page_size = anchor_host.page_size
        self.index_kpool = device_pool.index_kpool
        if not getattr(device_pool, "kpool_use_compress", False):
            raise ValueError("NPU HiCache indexer requires compressed KPool")
        if (
            self.index_kpool <= 1
            or self.page_size <= 0
            or self.page_size != device_pool.page_size
            or self.page_size % self.index_kpool
        ):
            raise ValueError(
                "KV page_size must match the device and divide by index_kpool"
            )
        if getattr(anchor_host, "dcp_size", 1) != 1 or getattr(
            device_pool, "layer_shard_enabled", False
        ):
            raise NotImplementedError(
                "NPU HiCache indexer CP/layer sharding is unsupported"
            )
        self.mtp_draft_device_pools = tuple(
            getattr(anchor_host, "mtp_draft_device_pools", ())
            if mtp_draft_device_pools is None
            else mtp_draft_device_pools
        )
        pools = (device_pool, *self.mtp_draft_device_pools)
        if len({id(pool) for pool in pools}) != len(pools):
            raise ValueError("MTP indexer device pools must be distinct")
        for draft in self.mtp_draft_device_pools:
            if getattr(draft, "layer_shard_enabled", False):
                raise NotImplementedError(
                    "NPU HiCache indexer CP/layer sharding is unsupported"
                )
            if (
                draft.layer_num != 1
                or draft.page_size != self.page_size
                or draft.index_kpool != self.index_kpool
                or not getattr(draft, "kpool_use_compress", False)
            ):
                raise ValueError(
                    "MTP indexer requires one draft layer with matching geometry"
                )
        self.slots_per_page = self.page_size // self.index_kpool
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num
        if self.size != self.page_num * self.page_size:
            raise ValueError("Anchor capacity is not a whole number of KV pages")
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        native = _native_index_buffers(device_pool)
        self.target_device_layer_num = device_pool.layer_num
        self.target_layer_num = len(native.index_layer_indices)
        self._host_layer_by_device = {
            layer: slot for slot, layer in enumerate(native.index_layer_indices)
        }
        self._host_layer_by_device.update(
            {
                device_pool.layer_num + depth: self.target_layer_num + depth
                for depth in range(len(self.mtp_draft_device_pools))
            }
        )
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)
        self.layout = "layer_first"
        self.device = "cpu"
        self.dtype = torch.uint8
        self.pin_memory = pin_memory
        self.can_use_write_back_jit = False
        self.allocator = None
        self._destroyed = False
        self._components = []
        self.bytes_per_layer = 0
        for name in ("index_k", "index_scales"):
            component = getattr(native, name)
            buffers = None if component is None else tuple(component.unbind(0))
            draft_buffers = [
                getattr(_native_index_buffers(draft), name)
                for draft in self.mtp_draft_device_pools
            ]
            draft_buffers = [
                None if value is None else tuple(value.unbind(0))
                for value in draft_buffers
            ]
            if any((value is None) != (buffers is None) for value in draft_buffers):
                raise ValueError(f"MTP indexer {name} components differ")
            if buffers is None:
                if name == "index_k":
                    raise ValueError("NPU index_k_buffer is missing")
                continue
            if len(buffers) != self.target_layer_num or not self.target_layer_num:
                raise ValueError(f"{name} must contain every logical FULL layer")
            if any(len(value) != 1 for value in draft_buffers):
                raise ValueError(f"MTP indexer {name} requires one layer per draft")
            buffers = [*buffers, *(value[0] for value in draft_buffers)]
            reference = buffers[0]
            if reference.device.type != "cpu" and not pin_memory:
                raise ValueError(
                    "NPU indexer asynchronous copies require pinned Host memory"
                )
            if reference.ndim < 2 or reference.shape[1] != self.slots_per_page:
                raise ValueError(
                    f"{name} must contain one compressed index page per allocator page"
                )
            if not reference.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            views = []
            for buffer in buffers:
                if (
                    buffer.shape != reference.shape
                    or buffer.dtype != reference.dtype
                    or not buffer.is_contiguous()
                    or buffer.device != reference.device
                ):
                    raise ValueError(f"{name} layer layouts differ")
                views.append(buffer.view(buffer.shape[0] * self.slots_per_page, -1))
            if self._components and views[0].shape[0] != self.device_rows:
                raise ValueError("Indexer key and scale page capacities differ")
            self.device_rows = views[0].shape[0]
            width = views[0].shape[1]
            byte_count = self.slots_per_page * width * reference.element_size()
            self._components.append((views, self.bytes_per_layer, byte_count, width))
            self.bytes_per_layer += (byte_count + 7) // 8 * 8
        if (
            not self._components[0][0][0].is_floating_point()
            and len(self._components) == 1
        ):
            raise ValueError("Quantized NPU indexer keys require index_scale_buffer")
        self.bytes_per_page = self.layer_num * self.bytes_per_layer
        self.size_per_token = self.get_size_per_token()
        self.allocated_bytes = self.page_num * self.bytes_per_page
        available_bytes = host_memory_budget_bytes(self.allocated_bytes)
        if self.allocated_bytes > available_bytes:
            raise ValueError(
                f"NPU HiCache indexer needs {self.allocated_bytes} additional host bytes; "
                f"rank budget has {available_bytes}. KV backing and Mamba are separate."
            )
        self.kv_buffer = self.init_kv_buffer()
        logger.info(
            "NPU HiCache indexer allocates %d host bytes (%d compressed slots/KV page); "
            "anchor KV backing is reused, Mamba budget is separate.",
            self.allocated_bytes,
            self.slots_per_page,
        )

    def get_size_per_token(self):
        return self.bytes_per_page / self.page_size

    def init_kv_buffer(self):
        return torch.zeros(
            (self.layer_num, self.page_num, self.bytes_per_layer),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def _require_live(self):
        if self._destroyed:
            raise RuntimeError("NPU HiCache indexer pool is destroyed")

    def _pages(self, indices, capacity):
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("HiCache indices must be a one-dimensional integer tensor")
        if indices.device.type != "cpu":
            raise ValueError("Stage HiCache indices on CPU once before layer transfers")
        rows = indices.to(device="cpu", dtype=torch.int64)
        if rows.numel() % self.page_size:
            raise ValueError(
                "NPU indexer cannot restore a partial KV page or live tail"
            )
        rows = rows.reshape(-1, self.page_size)
        starts = rows[:, 0]
        if (
            torch.any(starts < 0)
            or torch.any(starts + self.page_size > capacity)
            or torch.any(starts % self.page_size)
            or not torch.equal(rows, starts[:, None] + torch.arange(self.page_size))
        ):
            raise ValueError(
                "HiCache indices must contain ordered, aligned whole pages"
            )
        pages = starts // self.page_size
        if pages.unique().numel() != pages.numel():
            raise ValueError("Duplicate KV pages in an indexer transfer")
        return pages

    def _transfer_rows(self, device_pool, host_indices, device_indices, io_backend):
        self._require_live()
        if device_pool is not self.device_pool:
            raise ValueError("NPU indexer sidecar belongs to a different device pool")
        if io_backend not in ("kernel", "direct", "kernel_ascend"):
            raise ValueError(f"Unsupported NPU indexer IO backend: {io_backend}")
        host_pages = self._pages(host_indices, self.size)
        device_pages = self._pages(device_indices, self.device_rows * self.index_kpool)
        if host_pages.numel() != device_pages.numel():
            raise ValueError("Host and device page counts differ")
        if (device_pages == 0).any():
            raise ValueError("Device page zero is reserved for padding")
        return host_pages, device_pages * self.slots_per_page

    def _copy_runs(self, host_pages, locations, byte_count):
        runs = []
        contiguous_host_pages = byte_count == self.bytes_per_layer
        for host_page, location in zip(
            host_pages.tolist(), locations.tolist(), strict=True
        ):
            if (
                runs
                and contiguous_host_pages
                and host_page == runs[-1][0] + runs[-1][2]
                and location == runs[-1][1] + runs[-1][2] * self.slots_per_page
            ):
                start_page, start_row, count = runs[-1]
                runs[-1] = (start_page, start_row, count + 1)
            else:
                runs.append((host_page, location, 1))
        return tuple(runs)

    def _copy_layer(self, layer_id, plans, *, to_host):
        for (buffers, offset, byte_count, width), runs in zip(
            self._components, plans, strict=True
        ):
            buffer = buffers[layer_id]
            for host_page, device_row, page_count in runs:
                host_bytes = self.kv_buffer[
                    layer_id,
                    host_page : host_page + page_count,
                    offset : offset + byte_count,
                ]
                if not host_bytes.is_contiguous():
                    raise RuntimeError(
                        "Indexer copy run must use contiguous final Host storage"
                    )
                host_value = host_bytes.view(buffer.dtype).view(-1, width)
                device_value = buffer[
                    device_row : device_row + page_count * self.slots_per_page
                ]
                destination, source = (
                    (host_value, device_value)
                    if to_host
                    else (device_value, host_value)
                )
                destination.copy_(source, non_blocking=True)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_pages, locations = self._transfer_rows(
            device_pool, host_indices, device_indices, io_backend
        )
        plans = [
            self._copy_runs(host_pages, locations, component[2])
            for component in self._components
        ]
        for layer in range(self.layer_num):
            self._copy_layer(layer, plans, to_host=True)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        self.prepare_load_to_device(
            device_pool, host_indices, device_indices, io_backend, is_draft=is_draft
        )(layer_id)

    def prepare_load_to_device(
        self, device_pool, host_indices, device_indices, io_backend, *, is_draft=False
    ):
        """Snapshot CPU metadata once before L2 enters its per-layer copy loop."""
        if is_draft:
            draft_depth = next(
                (
                    depth
                    for depth, pool in enumerate(self.mtp_draft_device_pools)
                    if pool is device_pool
                ),
                None,
            )
            if draft_depth is None:
                raise ValueError(
                    "NPU indexer sidecar belongs to a different draft device pool"
                )
            first_layer = self.target_device_layer_num + draft_depth
            last_layer = first_layer + 1
        else:
            if device_pool is not self.device_pool:
                raise ValueError(
                    "NPU indexer sidecar belongs to a different device pool"
                )
            first_layer, last_layer = 0, self.target_device_layer_num
        host_pages, locations = self._transfer_rows(
            self.device_pool, host_indices, device_indices, io_backend
        )
        plans = tuple(
            self._copy_runs(host_pages, locations, component[2])
            for component in self._components
        )

        def load_layer(layer_id):
            self._require_live()
            if not first_layer <= layer_id < last_layer:
                raise IndexError("NPU indexer layer is outside this pool")
            host_layer = self._host_layer_by_device.get(layer_id)
            if host_layer is None:
                raise IndexError("NPU indexer layer owns no index buffer")
            self._copy_layer(host_layer, plans, to_host=False)

        return load_layer

    def _page_index(self, index):
        self._require_live()
        index = int(index)
        if index < 0 or index >= self.size or index % self.page_size:
            raise ValueError(
                "Host data page must start at an aligned in-range KV index"
            )
        return index // self.page_size

    def get_data_page(self, index, flat=True):
        page = self.kv_buffer[:, self._page_index(index)]
        return page.contiguous().flatten() if flat else page

    def get_dummy_flat_data_page(self):
        self._require_live()
        return torch.zeros(self.bytes_per_page, dtype=self.dtype)

    def set_from_flat_data_page(self, index, data_page):
        page = self.get_data_page(index, flat=False)
        if data_page.dtype != self.dtype or data_page.numel() != self.bytes_per_page:
            raise ValueError("Serialized NPU indexer page has the wrong dtype or size")
        page.copy_(data_page.reshape_as(page))

    def get_hybrid_pool_buffer(self):
        self._require_live()
        return [self.kv_buffer]

    def get_page_buffer_meta(self, indices):
        """Return page-major lists of per-layer ranges, never one fictitious span."""
        self._require_live()
        pages = self._pages(indices, self.size)
        pointers = [
            self.kv_buffer[layer, int(page)].data_ptr()
            for page in pages
            for layer in range(self.layer_num)
        ]
        return pointers, [self.bytes_per_layer] * len(pointers)

    def is_stride_page_aligned(self, page_size_bytes=4096):
        self._require_live()
        return (
            self.bytes_per_layer % page_size_bytes == 0
            and self.kv_buffer.data_ptr() % page_size_bytes == 0
        )

    def alloc(self, need_size):
        raise RuntimeError("INDEXER uses SidecarPoolSpec indices from the KV anchor")

    def free(self, indices):
        raise RuntimeError("Only the KV anchor may release shared INDEXER page indices")

    def available_size(self):
        self._require_live()
        return self.anchor_host.available_size()

    def clear(self):
        self._require_live()

    def destroy(self):
        self._destroyed = True
        self.kv_buffer = None
