from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sglang.srt.disaggregation import decode
from sglang.srt.disaggregation.mooncake.buffers import (
    BufferKeys,
    BufferKind,
    MooncakeBufferKey,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    TransferInfo,
)
from sglang.srt.disaggregation.mooncake.transport import MooncakeTransport
from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUDSATokenToKVPool,
    NPUMLATokenToKVPool,
)
from sglang.srt.mem_cache import kv_cache_configurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_layer_staging_registration_keeps_buffer_keys_and_drains_before_deregister():
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.kv_args = SimpleNamespace(
        kv_data_ptrs=[100, 200, 0],
        kv_data_lens=[16, 32, 0],
        aux_data_ptrs=[300],
        aux_data_lens=[8],
        state_data_ptrs=[[100]],
        state_data_lens=[[16]],
        gpu_id=0,
    )
    mgr._local_keys = BufferKeys.for_buffers([100, 200, 0], [300], [[100]])
    mgr._layer_staged_host_regions = {(200, 32)}
    mgr._layer_staging = Mock(address=400, window_bytes=64)
    mgr._layer_leases = {19: object()}
    engine = Mock()
    mgr.transport = MooncakeTransport(engine=engine)

    registrations = {r.key: (r.ptr, r.length) for r in mgr._kv_args_registrations()}
    assert registrations[mgr._local_keys.kv[0]] == (100, 16)
    assert mgr._local_keys.kv[1] not in registrations
    assert registrations[mgr._local_keys.state[0][0]] == (100, 16)
    assert registrations[MooncakeBufferKey(BufferKind.STAGING, -1)] == (400, 64)
    mgr.register_buffer_to_engine()
    engine.batch_register.assert_called_once_with([100, 300, 400], [16, 8, 64])
    with pytest.raises(RuntimeError, match="layer leases"):
        mgr.deregister_buffer_to_engine()
    engine.batch_deregister.assert_not_called()
    mgr._layer_leases.clear()
    mgr.deregister_buffer_to_engine()
    mgr._layer_staging.drain.assert_called_once_with()
    assert set(engine.batch_deregister.call_args.args[0]) == {100, 300, 400}


@pytest.mark.parametrize("manifest", [b"", b"layer-lease"])
def test_layer_manifest_extends_existing_mooncake_wire_message(manifest):
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.local_ip, mgr.rank_port = "127.0.0.1", 1234
    mgr.get_session_id = lambda: "decode"
    receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
    receiver._layer_manifest = manifest
    receiver._layer_published_peers = set()
    peer = {"rank_ip": "127.0.0.2", "rank_port": 4321}
    message = mgr.build_receiver_metadata_message(
        bootstrap_room=19,
        required_dst_info_num=1,
        kv_indices=np.array([5], dtype=np.int32),
        aux_index=3,
        state_indices=None,
        decode_prefix_len=512,
        is_dummy=False,
        device_kv_indices=np.array([7], dtype=np.int32),
    )
    with patch.object(
        MooncakeKVReceiver.__bases__[0], "_send_bootstrap_message", return_value=True
    ) as send:
        assert receiver._send_bootstrap_message(
            bootstrap_info=peer, message=message, operation="send_metadata"
        )
    wire = send.call_args.kwargs["message"]
    assert len(wire) == len(message) + bool(manifest)
    parsed = TransferInfo.from_zmq(wire)
    assert parsed.layer_manifest == manifest
    assert parsed.decode_prefix_len == 512
    assert parsed.dst_kv_indices.tolist() == [5]
    assert parsed.dst_device_kv_indices.tolist() == [7]
    assert receiver._layer_published_peers == (
        {("127.0.0.2", 4321)} if manifest else set()
    )


def test_layer_transfer_timeout_retains_pages_until_receiver_confirms_drain():
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    lease = SimpleNamespace(drained=False)
    mgr._layer_leases = {19: lease}
    request = SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=19), kv_receiver=SimpleNamespace(kv_mgr=mgr)
    )
    queue = decode.DecodeTransferQueue.__new__(decode.DecodeTransferQueue)
    queue._is_host_destination = Mock(return_value=False)
    queue._deferred_releases = Mock()
    queue._deferred_releases.items.return_value = [(request, 10.0, 12.0, 3, 1)]
    queue.gloo_group = None
    queue._do_release = Mock()
    queue._observe_deferred_kv_release = Mock()
    with patch.object(decode.dist, "get_world_size", return_value=1):
        with patch.object(decode.time, "monotonic", return_value=11.0):
            queue.resolve_deferred_releases()
        queue._do_release.assert_not_called()
        with patch.object(decode.time, "monotonic", return_value=13.0):
            with pytest.raises(RuntimeError, match="did not confirm drain"):
                queue.resolve_deferred_releases()
            queue._do_release.assert_not_called()
            lease.drained = True
            queue.resolve_deferred_releases()
    queue._do_release.assert_called_once_with(request, 3)


@pytest.mark.parametrize(
    "index_kpool,expected", [(4, NPUMLATokenToKVPool), (1, NPUDSATokenToKVPool)]
)
def test_glm5next_hybrid_npu_full_attention_leaf(index_kpool, expected):
    hf_config = SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        index_topk=32,
        index_kpool=index_kpool,
    )
    kv = SimpleNamespace(
        use_mla_backend=True, model_config=SimpleNamespace(hf_config=hf_config)
    )
    with patch.object(kv_cache_configurator, "_is_npu", True):
        leaf = (
            kv_cache_configurator.KVCacheConfigurator._hybrid_full_attention_pool_class(
                kv, mha_pool_class=object, mla_pool_class=object, dsa_pool_class=object
            )
        )
    assert leaf is expected
