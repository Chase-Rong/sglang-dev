"""Host memory registration and row copies for Ascend HiSparse."""

from dataclasses import dataclass
from math import prod
from typing import Optional, Sequence, Tuple

import torch


def host_ops():
    from sgl_kernel_npu.hisparse import load_host

    return load_host()


@dataclass
class RegisteredHostRegion:
    """Host tensor plus its NPU-visible registered address.

    Explicit close requires drained device and captured-graph users.
    Tensor aliases must not outlive close. There is deliberately no destructor
    that could unregister memory while asynchronous users still reference it.
    """

    tensor: torch.Tensor
    host_ptr: int
    device_ptr: int
    device_id: int
    name: str
    closed: bool = False

    def _require_live(self) -> None:
        if self.closed:
            raise RuntimeError("registered Host region is closed")

    @property
    def nbytes(self) -> int:
        self._require_live()
        return self.tensor.numel() * self.tensor.element_size()

    def view(self, shape: Sequence[int]) -> torch.Tensor:
        self._require_live()
        view = self.tensor.view(tuple(shape))
        if view.data_ptr() != self.host_ptr:
            raise RuntimeError("registered region view is not base-aligned")
        return view


def copy_rows_to_registered_host(
    src: torch.Tensor,
    dst_region: RegisteredHostRegion,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    block_dim: int = 8,
) -> torch.Tensor:
    """Copy device rows into a registered Host region.

    ``src_index``, ``dst_index`` and ``valid_mask`` must use the device
    expected by the compiled kernel.  The destination pointer is explicitly
    passed as the NPU-visible address; the CPU process pointer never crosses
    the PD control protocol.
    """

    dst_region._require_live()
    return unidex_copy_inplace(
        src=src,
        dst=dst_region.tensor,
        src_index=src_index,
        dst_index=dst_index,
        valid_mask=valid_mask,
        src_address_ndims=1,
        dst_address_ndims=1,
        block_dim=block_dim,
        dst_ptr=dst_region.device_ptr,
    )


def _infer_rows_and_block_bytes(
    tensor: torch.Tensor, address_ndims: int, name: str
) -> Tuple[int, int]:
    if address_ndims <= 0 or address_ndims >= tensor.dim():
        raise ValueError(
            f"{name}_address_ndims must be in [1, {tensor.dim() - 1}], "
            f"got {address_ndims}"
        )

    rows = prod(tensor.shape[:address_ndims])
    block_elements = prod(tensor.shape[address_ndims:])
    return int(rows), int(block_elements * tensor.element_size())


def unidex_copy_inplace(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    valid_mask: torch.Tensor,
    src_address_ndims: int,
    dst_address_ndims: int,
    block_dim: int = 8,
    src_ptr: Optional[int] = None,
    dst_ptr: Optional[int] = None,
) -> torch.Tensor:
    """Copy selected logical rows from ``src`` into ``dst`` in place.

    ``src_ptr`` and ``dst_ptr`` may override the Tensor addresses with
    device-visible shared-memory addresses. The caller owns those allocations
    and must keep them alive until work on the current NPU stream completes.
    """
    src_rows, src_block_bytes = _infer_rows_and_block_bytes(
        src, src_address_ndims, "src"
    )
    dst_rows, dst_block_bytes = _infer_rows_and_block_bytes(
        dst, dst_address_ndims, "dst"
    )
    if src_block_bytes != dst_block_bytes:
        raise ValueError(
            "src and dst logical rows must have the same byte size, got "
            f"{src_block_bytes} and {dst_block_bytes}"
        )
    if (
        src_index.numel() != dst_index.numel()
        or src_index.numel() != valid_mask.numel()
    ):
        raise ValueError(
            "src_index, dst_index, and valid_mask must have the same length"
        )
    if src.dtype != dst.dtype:
        raise ValueError(
            f"src and dst must have the same dtype, got {src.dtype} and {dst.dtype}"
        )

    host_ops().unidex_copy(
        src,
        dst,
        src_index,
        dst_index,
        valid_mask,
        src_rows,
        dst_rows,
        src_block_bytes,
        src_index.numel(),
        block_dim,
        src_ptr,
        dst_ptr,
    )
    return dst
