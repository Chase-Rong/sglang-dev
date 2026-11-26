"""MemFabric and HiSparse DMA round trip on a CANN 9.3 Ascend 950 device.

Run with one device visible and at least 1 GiB of host memory for the pool.
"""

import pytest
import torch


def test_local_production_dma():
    import torch_npu

    device = torch.device("npu", torch_npu.npu.current_device())
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops
    from sglang.srt.hardware_backend.npu.memfabric_host_kv import MemFabricHostKvPool

    pool = MemFabricHostKvPool(device.index, required_bytes=256 * 512 * 2)
    graph = None
    try:
        region = pool.allocate_component(
            "kv", (256, 512), torch.bfloat16, page_size=128
        ).region
        assert pool.physical_bytes == 1 << 30
        rows = torch.tensor([128, 129, 130, 7, 8, 18, 0, -1, 256], device=device)
        plan, _ = pool.plan_prefill_rows(rows, rows[:0], 256)
        valid = torch.tensor([128, 129, 130, 7, 8, 18])
        pattern = (torch.arange(256 * 512).view(256, 512) % 97).bfloat16() / 128
        source = pattern.to(device)
        restored = torch.zeros_like(source)
        copy_stream = torch.npu.Stream(device=device)
        compute_stream = torch.npu.current_stream()
        with pytest.raises(RuntimeError, match="drain"):
            pool.close()
        for shift in (0, 1, 2):
            source.copy_((pattern + shift).to(device))
            ready = torch.npu.Event()
            ready.record(compute_stream)
            with torch.npu.stream(copy_stream):
                copy_stream.wait_event(ready)
                pool.copy_prefill_to_host("kv", source, plan)
                saved = torch.npu.Event()
                saved.record(copy_stream)
            compute_stream.wait_event(saved)
            pool.copy_prefill_from_host("kv", restored, plan)
            torch.npu.synchronize()
            torch.testing.assert_close(
                restored.cpu()[valid], (pattern + shift)[valid], atol=0, rtol=0
            )
            torch.testing.assert_close(
                region.tensor[valid], (pattern + shift)[valid], atol=0, rtol=0
            )
            untouched = torch.ones(256, dtype=torch.bool)
            untouched[valid] = False
            assert not restored.cpu()[untouched].any()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            pool.copy_prefill_to_host("kv", source, plan)
            pool.copy_prefill_from_host("kv", restored, plan)
        source.copy_((pattern + 3).to(device))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            restored.cpu()[valid], (pattern + 3)[valid], atol=0, rtol=0
        )
        for bad in (
            torch.tensor([[256, 1]]),
            torch.tensor([[1, 256]]),
            torch.tensor([[0, 1]]),
        ):
            with pytest.raises(RuntimeError, match="outside"):
                pool.copy_prefill_from_host("kv", restored, bad)
        with pytest.raises(RuntimeError, match="live MemFabric"):
            native_ops().prefill_dma_copy(
                restored, region.tensor, region.device_ptr + 1, plan, False
            )
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        pool.close(drained=True)
