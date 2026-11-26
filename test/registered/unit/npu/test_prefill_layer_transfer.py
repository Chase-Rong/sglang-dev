import sys

import pytest

from sglang.srt.hardware_backend.npu.hisparse import layer_transfer as transfer
from sglang.srt.hardware_backend.npu.hisparse import prefill_offload as protocol
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def inputs():
    token = protocol.LayerSlotToken(1, "req", 4, 7)
    descriptors, sources, destinations = [], {}, {}
    for index, component in enumerate(protocol.KvComponent):
        descriptors.append(
            protocol.PdComponentDescriptor(
                request_id="req",
                logical_layer=4,
                generation=7,
                component=component,
                source_kind=protocol.PdSourceKind.NPU_REGISTERED,
                source_slot=1,
                source_page_ids=(2, 0),
                destination_page_ids=(1, 3),
                valid_range=protocol.KvPageRange(1, 3),
                bytes_per_page=16,
            )
        )
        sources[component] = transfer.RegisteredLayerRegion(
            1000 + 100 * index,
            64,
            16,
            f"source-{index}",
            protocol.PdSourceKind.NPU_REGISTERED,
            "bf16",
            object(),
        )
        destinations[component] = transfer.RegisteredLayerRegion(
            2000 + 100 * index,
            64,
            16,
            f"destination-{index}",
            protocol.PdSourceKind.NPU_REGISTERED
            if component == protocol.KvComponent.INDEXER_K
            else protocol.PdSourceKind.HOST_REGISTERED,
            "bf16",
            object(),
        )
    return protocol.LayerReadyRecord(token, tuple(descriptors)), sources, destinations


def test_transfer_copies_only_partial_page_range_and_independent_components():
    record, sources, destinations = inputs()
    blocks = transfer.build_layer_transfer_blocks(
        record, sources, destinations, page_size=2
    )
    assert blocks == [
        (1040, 2024, 8),
        (1000, 2048, 8),
        (1140, 2124, 8),
        (1100, 2148, 8),
        (1240, 2224, 8),
        (1200, 2248, 8),
    ]


def test_nope_layer_transfer_accepts_registered_latent_without_rope():
    record, sources, destinations = inputs()
    keep = {protocol.KvComponent.LATENT_K, protocol.KvComponent.INDEXER_K}
    record = protocol.LayerReadyRecord(
        record.token,
        tuple(entry for entry in record.components if entry.component in keep),
    )
    sources = {key: value for key, value in sources.items() if key in keep}
    destinations = {key: value for key, value in destinations.items() if key in keep}
    blocks = transfer.build_layer_transfer_blocks(
        record, sources, destinations, page_size=2
    )
    assert blocks == [
        (1040, 2024, 8),
        (1000, 2048, 8),
        (1240, 2224, 8),
        (1200, 2248, 8),
    ]


def test_layer_transfer_rejects_publication_missing_registered_component():
    record, sources, destinations = inputs()
    record = protocol.LayerReadyRecord(record.token, record.components[:-1])
    with pytest.raises(ValueError, match="components do not match"):
        transfer.build_layer_transfer_blocks(record, sources, destinations, page_size=2)


def test_transfer_rejects_unregistered_bounds_or_wrong_destination_kind():
    record, sources, destinations = inputs()
    component = protocol.KvComponent.LATENT_K
    destinations[component] = transfer.RegisteredLayerRegion(
        2000,
        32,
        16,
        "short-region",
        protocol.PdSourceKind.HOST_REGISTERED,
        "bf16",
        object(),
    )
    with pytest.raises(ValueError, match="exceeds"):
        transfer.build_layer_transfer_blocks(record, sources, destinations, page_size=2)
    destinations[component] = sources[component]
    with pytest.raises(ValueError, match="address-space"):
        transfer.build_layer_transfer_blocks(record, sources, destinations, page_size=2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
