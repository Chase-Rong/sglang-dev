from dataclasses import replace
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.hicache_indexer_mapping import (
    get_prefill_device_pages,
)
from sglang.srt.hardware_backend.npu.hicache_indexer_pool import (
    NPUIndexerPoolHost,
    npu_indexer_host_bytes_per_layer_page,
)
from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
    DecodeKvMemoryBudget,
)
from sglang.srt.hardware_backend.npu.hisparse.hicache_offload import OffloadPageBinding
from sglang.srt.mem_cache.device_pool_info import MLABufferInfo, MLASplitBuffers
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_host_page_table_preserves_attention_offsets_and_zeroes_missing_pages():
    host = SimpleNamespace(
        page_size=512,
        page_num=8,
        device_pool=SimpleNamespace(physical_page_size=128),
        backing=SimpleNamespace(pool=SimpleNamespace()),
    )
    binding = OffloadPageBinding(host, "cpu", 4)
    binding.device_to_host.copy_(torch.tensor([0, 6, 0, 2]))
    physical_pages = torch.tensor(
        [[4, 5, 6, 7, 8, 11, 14, 16, -1, 3]], dtype=torch.int32
    )
    expected = torch.tensor([[24, 25, 26, 27, 0, 0, 10, 0, 0, 0]], dtype=torch.int32)
    torch.testing.assert_close(binding.host_table(physical_pages), expected)
    replay = torch.full((2, 12), -1, dtype=torch.int32)
    binding.fill_host_table(physical_pages, replay)
    torch.testing.assert_close(replay[:1, :10], expected)
    assert not replay[1].any() and not replay[:, 10:].any()


def test_continuing_prefill_preserves_request_order_and_partial_final_page():
    table = torch.zeros(3, 1536, dtype=torch.int32)
    table[1, ::512] = torch.tensor([2560, 1024, 3072])
    table[2, 0] = 2048
    batch = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        tree_cache=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=512)
        ),
    )
    pages = get_prefill_device_pages(
        batch, torch.tensor([2, 1]), torch.tensor([2, 1025])
    )
    torch.testing.assert_close(pages, torch.tensor([4, 5, 2, 6]))


def test_indexer_sidecar_copies_one_compressed_page_for_each_kv_page(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.hardware_backend.npu.hicache_indexer_pool.host_memory_budget_bytes",
        lambda requested_bytes: requested_bytes,
    )
    keys = torch.arange(4 * 128 * 4, dtype=torch.float32).view(1, 4, 128, 1, 4)
    buffers = MLASplitBuffers(
        k=None,
        v=None,
        index_k=keys,
        index_scales=None,
        index_layer_indices=(0,),
        k_width=512,
        v_width=0,
        index_width=4,
    )
    info = MLABufferInfo(
        buffers=buffers,
        data_ptrs=None,
        layer_indices=(0,),
        device_layer_num=1,
        page_size=512,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        kv_cache_dim=512,
        dtype=keys.dtype,
        device=keys.device,
    )
    pool = SimpleNamespace(
        page_size=512,
        index_kpool=4,
        kpool_use_compress=True,
        start_layer=0,
        end_layer=1,
        layer_num=1,
        get_device_pool_infos=lambda: (
            SimpleNamespace(buffer_info=info, layer_ids=(0,)),
        ),
    )
    anchor = SimpleNamespace(page_size=512, size=2048, page_num=4, binding=None)
    sidecar = NPUIndexerPoolHost(pool, anchor, pin_memory=False)
    assert npu_indexer_host_bytes_per_layer_page(pool) == 128 * 4 * keys.element_size()
    source = keys[0, 3].clone()
    sidecar.backup_from_device_all_layer(
        pool, torch.arange(512, 1024), torch.arange(1536, 2048), "kernel_ascend"
    )
    sidecar.load_to_device_per_layer(
        pool, torch.arange(512, 1024), torch.arange(1024, 1536), 0, "kernel_ascend"
    )
    torch.testing.assert_close(keys[0, 2], source)


def test_budget_uses_allocator_pages_for_kv_and_attention_pages_for_the_planner():
    budget = DecodeKvMemoryBudget(
        layers=2,
        latent_dim=512,
        rope_dim=0,
        index_dim=128,
        element_bytes=2,
        topk=2051,
        resident_capacity=2176,
        host_bytes=2**32,
        host_index_kpool=4,
        host_index_sidecar=True,
        execution_abi="shared_v3",
    )
    uncompressed = replace(budget, host_index_kpool=1, host_index_sidecar=False)
    assert budget.fixed_bytes(2, 512) == uncompressed.fixed_bytes(2, 128)
    assert budget.device_bytes_per_token == 2 * 128 * 2 // 4
    # Two live allocator pages and page zero, each with 128 compressed keys per layer.
    assert budget.host_indexer_bytes(1024, 512) == 3 * 2 * 128 * 128 * 2
