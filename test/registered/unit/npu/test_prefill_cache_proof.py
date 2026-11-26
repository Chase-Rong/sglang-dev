"""CPU tree proofs: genuine ancestors and component-valid KDA boundaries."""

from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import TreeComponent
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tree_slots import slot_of, tree_slots_of

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def tree_fixture():
    params = CacheInitParams(
        False,
        None,
        None,
        512,
        tree_slots=tree_slots_of(
            (ComponentType.FULL, ComponentType.MAMBA), checkpoint_interval=512
        ),
    )
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.enable_session_radix_cache = False
    full = FullComponent(cache, params, slot_of(params, ComponentType.FULL))
    mamba = MambaComponent.__new__(MambaComponent)
    TreeComponent.__init__(mamba, cache, params, slot_of(params, ComponentType.MAMBA))
    mamba.host_only_checkpoint = False
    mamba.mamba_checkpoint_grid = 512
    cache.tree_core = UnifiedTreeCore(
        params, {ComponentType.FULL: full, ComponentType.MAMBA: mamba}
    )
    cache._components_tuple = (full, mamba)
    cache.linker = SimpleNamespace(
        private_load_indices={"rid": torch.arange(3000, 3512)}
    )

    def add(parent, tokens, device_rows, checkpoint):
        node = cache.tree_core._new_node()
        node.parent = parent
        node.key = RadixKey(array("q", tokens), extra_key="model", cache_salt="salt")
        node.component_data[ComponentType.FULL].value = device_rows
        node.component_data[ComponentType.MAMBA].value = (
            torch.tensor([1]) if checkpoint else None
        )
        parent.children[node.key.child_key(512)] = node
        return node

    first = add(cache.tree_core.root_node, range(512), torch.arange(128, 640), True)
    final = add(first, range(512, 1024), torch.arange(1024, 1536), False)
    req = SimpleNamespace(
        rid="rid",
        extra_key="model",
        cache_salt="salt",
        last_node=final.id,
        prefix_len=1536,
        origin_input_ids=list(range(1536)),
        output_ids=[],
    )
    return cache, req, first, final


def test_public_proof_excludes_private_tail_and_full_rows_beyond_kda_checkpoint():
    cache, req, first, final = tree_fixture()
    rows = torch.cat(
        (torch.arange(128, 640), torch.arange(1024, 1536), torch.arange(3000, 3512))
    )
    proof = cache.prefill_cache_prefix_proof(req, rows)
    assert proof.shared_prefix_len == 512
    assert proof.device_rows == tuple(range(128, 640))
    assert proof.token_ids == tuple(range(512))
    # A valid final KDA checkpoint expands the public boundary to 1024.
    final.component_data[ComponentType.MAMBA].value = torch.tensor([2])
    proof = cache.prefill_cache_prefix_proof(req, rows)
    assert proof.shared_prefix_len == 1024
    rows.zero_()
    assert proof.device_rows == tuple(range(128, 640)) + tuple(range(1024, 1536))


@pytest.mark.parametrize("wrong", ["rows", "tokens", "namespace", "salt"])
def test_public_proof_rejects_wrong_admission_source(wrong):
    cache, req, first, final = tree_fixture()
    rows = torch.cat(
        (torch.arange(128, 640), torch.arange(1024, 1536), torch.arange(3000, 3512))
    )
    if wrong == "rows":
        rows[0] = 999
    elif wrong == "tokens":
        req.origin_input_ids[0] = 999
    elif wrong == "namespace":
        req.extra_key = "another"
    else:
        req.cache_salt = "another"
    with pytest.raises(ValueError, match="cache proof"):
        cache.prefill_cache_prefix_proof(req, rows)


def test_public_proof_requires_complete_512_group_and_real_kda_state():
    cache, req, first, final = tree_fixture()
    req.prefix_len = 384
    rows = torch.arange(128, 512)
    assert cache.prefill_cache_prefix_proof(req, rows) is None
    req.prefix_len = 512
    first.component_data[ComponentType.MAMBA].value = None
    assert cache.prefill_cache_prefix_proof(req, torch.arange(128, 640)) is None


def test_forward_batch_samples_proof_from_real_req_kv_row(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.sampling.sampling_params import SamplingParams

    cache, source, first, final = tree_fixture()
    req = Req("proof", "", array("q", range(1024)), SamplingParams(max_new_tokens=1))
    req.extra_key, req.cache_salt = "model", "salt"
    req.kv.req_pool_idx = 1
    req.last_node, req.prefix_len = final.id, 1024
    table = torch.zeros(2, 1024, dtype=torch.int64)
    table[1] = torch.cat((torch.arange(128, 640), torch.arange(1024, 1536)))
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        tree_cache=cache,
        reqs=[req],
        req_to_token_pool=SimpleNamespace(req_to_token=table),
    )
    with envs.SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD.override(True):
        proofs = ForwardBatch._prefill_cache_prefix_proofs(batch)
    assert proofs[0].shared_prefix_len == 512
    table.zero_()
    assert proofs[0].device_rows == tuple(range(128, 640))


@pytest.mark.parametrize(
    "mode", ["DECODE", "IDLE", "PREBUILT", "TARGET_VERIFY", "DRAFT_EXTEND_V2"]
)
def test_non_prefill_does_not_build_prefix_proofs(mode):
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import ForwardMode

    def unexpected_proof(*args):
        pytest.fail("A non-prefill forward requested a prefill cache proof")

    batch = SimpleNamespace(
        forward_mode=ForwardMode[mode],
        tree_cache=SimpleNamespace(prefill_cache_prefix_proof=unexpected_proof),
        reqs=[SimpleNamespace(kv=SimpleNamespace(req_pool_idx=1), prefix_len=128)],
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(2, 128)),
    )
    with envs.SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD.override(True):
        assert ForwardBatch._prefill_cache_prefix_proofs(batch) is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
