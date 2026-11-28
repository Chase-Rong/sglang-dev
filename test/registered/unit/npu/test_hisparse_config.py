import json
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from sglang.srt.hardware_backend.npu.hisparse import config as CONFIG
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def args(role="null", npu=None, **changes):
    values = dict(
        enable_hisparse=True,
        disaggregation_mode=role,
        hisparse_config={"npu": npu or {}},
        enable_hierarchical_cache=role != "prefill",
        disable_radix_cache=role == "prefill",
        disaggregation_transfer_backend="ascend",
        kv_cache_dtype="bfloat16",
        dsa_prefill_backend=None,
        dsa_decode_backend=None,
        page_size=512,
        speculative_algorithm=None,
        enable_mixed_chunk=False,
        disaggregation_decode_enable_offload_kvcache=False,
        hicache_ratio=None,
        hicache_size=0,
        hicache_host_memory_fraction=0.8,
        hicache_host_memory_mode="cache",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def budget():
    return {"decode_host_kv_bytes": 1 << 30}


def model(**changes):
    values = dict(
        index_topk=2048,
        index_kpool=4,
        index_kpool_compress=True,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def validate(server, config=None, hf=None, **changes):
    CONFIG.validate_npu_hisparse(
        server,
        config or CONFIG.resolve_npu_hisparse_config(server, {}),
        hf or model(),
        is_dsa=changes.get("is_dsa", True),
        is_v4=changes.get("is_v4", False),
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_new_and_legacy_inputs_resolve_to_one_immutable_configuration(explicit):
    npu = budget()
    if explicit:
        npu["dsa_offload_backend"] = "external_plan"
    server = args(npu=npu)
    actual = CONFIG.resolve_npu_hisparse_config(server, {})
    legacy = args(enable_hisparse=False, hisparse_config=None)
    assert CONFIG.resolve_npu_hisparse_config(legacy, actual.environment()) == actual
    assert actual.integrated and actual.prefill_layerwise_offload
    assert actual.backend == "external_plan"
    with pytest.raises(FrozenInstanceError):
        actual.device_buffer_size = 1


def test_flag_requires_explicit_positive_byte_budget_without_allocating():
    environment = {}
    with pytest.raises(ValueError, match="explicit positive.*Host byte budget"):
        CONFIG.resolve_npu_hisparse_config(args(), environment)
    assert environment == {}


def test_non_offload_npu_is_unchanged():
    server = args(enable_hisparse=False, hisparse_config=None)
    assert CONFIG.resolve_npu_hisparse_config(server, {}) is None
    assert (
        CONFIG.resolve_npu_hisparse_config(
            server, {"SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION": "1"}
        )
        is None
    )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_npu_hisparse_requires_integrated_service(role):
    with pytest.raises(ValueError, match="requires --disaggregation-mode null"):
        CONFIG.resolve_npu_hisparse_config(args(role, budget()), {})
    assert (
        CONFIG.resolve_npu_hisparse_config(
            args(role, enable_hisparse=False, hisparse_config=None), {}
        )
        is None
    )


@pytest.mark.parametrize("field", list(CONFIG.ENV_FIELDS))
def test_each_explicit_legacy_alias_accepts_equal_json_and_rejects_conflict(field):
    name, default = CONFIG.ENV_FIELDS[field]
    value = {
        "integrated": True,
        "prefill_layerwise_offload": True,
        "fused_sparse_attention": True,
        "decode_host_kv_bytes": 1 << 30,
    }.get(field, default)
    npu = {**budget(), field: value}
    actual = CONFIG.resolve_npu_hisparse_config(args(npu=npu), {name: str(value)})
    assert getattr(actual, field) == value
    if type(value) is bool:
        conflicting = not value
    elif type(value) in (int, float):
        conflicting = value + 1
    else:
        conflicting = value + "_different"
    with pytest.raises(ValueError, match=f"conflicts with explicit {name}"):
        CONFIG.resolve_npu_hisparse_config(args(npu=npu), {name: str(conflicting)})


def test_unset_legacy_defaults_do_not_override_json_and_export_is_explicit():
    server = args(npu={**budget(), "prefill_layer_slots": 3, "shared_host_kv": False})
    server.hisparse_config.update(device_buffer_size=8192, host_to_device_ratio=999)
    environment = {}
    actual = CONFIG.resolve_npu_hisparse_config(server, environment)
    assert not environment
    actual.apply_environment(environment)
    assert environment["SGLANG_NPU_FSA_RESIDENT_CAPACITY"] == "8192"
    assert environment["SGLANG_NPU_PREFILL_LAYER_SLOTS"] == "3"
    assert environment["SGLANG_NPU_SHARED_HOST_KV"] == "False"
    assert actual.decode_host_kv_bytes == 1 << 30
    assert CONFIG.resolve_npu_hisparse_config(server, environment) == actual


def test_cli_json_string_preserves_all_budget_inputs():
    server = args(
        npu={
            **budget(),
            "hicache_host_tokens": 131072,
            "prefill_layer_slots": 1,
        }
    )
    original = CONFIG.resolve_npu_hisparse_config(server, {})
    server.hisparse_config = json.dumps(server.hisparse_config)
    assert CONFIG.resolve_npu_hisparse_config(server, {}) == original
    assert original.hicache_host_tokens == 131072
    assert original.prefill_layer_slots == 1


def test_top_level_resident_input_conflicts_identify_both_sources():
    server = args(npu={**budget(), "device_buffer_size": 8192})
    server.hisparse_config["device_buffer_size"] = 4096
    with pytest.raises(ValueError, match="device_buffer_size conflicts"):
        CONFIG.resolve_npu_hisparse_config(server, {})
    server.hisparse_config.pop("device_buffer_size")
    with pytest.raises(ValueError, match="SGLANG_NPU_FSA_RESIDENT_CAPACITY"):
        CONFIG.resolve_npu_hisparse_config(
            server, {"SGLANG_NPU_FSA_RESIDENT_CAPACITY": "4096"}
        )


def test_npu_json_does_not_silently_enable_an_omitted_flag():
    with pytest.raises(ValueError, match="requires --enable-hisparse"):
        CONFIG.resolve_npu_hisparse_config(
            args(npu=budget(), enable_hisparse=False), {}
        )


@pytest.mark.parametrize("raw", ["{", "[]", "null", "1", {"npu": []}, {"npu": None}])
def test_malformed_json_and_non_object_config_fail(raw):
    with pytest.raises(ValueError, match="hisparse_config"):
        CONFIG.resolve_npu_hisparse_config(args(hisparse_config=raw), {})


@pytest.mark.parametrize(
    "npu,match",
    [
        ({"unknown": 1}, "Unknown"),
        ({"decode_host_kv_bytes": True}, "expected int"),
        ({"decode_host_kv_bytes": -1}, "nonnegative"),
        ({"device_buffer_size": 0}, "positive"),
        ({"prefill_layer_slots": 0}, "positive"),
        ({"prefill_layer_slots": 1.5}, "expected int"),
        ({"shared_host_kv": "false"}, "expected bool"),
        ({"dsa_offload_backend": "cuda"}, "external_plan"),
        ({"fused_sparse_attention_mode": "reference"}, "requires.*fused"),
        ({"integrated": False}, "integrated NPU HiSparse"),
        ({"fused_sparse_attention": False}, "integrated NPU HiSparse"),
    ],
)
def test_invalid_fields_are_rejected_before_export(npu, match):
    with pytest.raises(ValueError, match=match):
        CONFIG.resolve_npu_hisparse_config(args(npu={**budget(), **npu}), {})


@pytest.mark.parametrize(
    "name,value",
    [
        ("SGLANG_NPU_SHARED_HOST_KV", "maybe"),
        ("SGLANG_NPU_DECODE_HOST_KV_BYTES", "lots"),
    ],
)
def test_invalid_legacy_values_are_errors_not_silent_defaults(name, value):
    with pytest.raises(ValueError):
        CONFIG.resolve_npu_hisparse_config(args(npu=budget()), {name: value})


@pytest.mark.parametrize("legacy", [False, True])
def test_other_backend_is_rejected_at_startup(legacy):
    npu = budget()
    env = {}
    if legacy:
        env["SGLANG_NPU_DSA_OFFLOAD_BACKEND"] = "lim_copy_sfa"
    else:
        npu["dsa_offload_backend"] = "lim_copy_sfa"
    with pytest.raises(ValueError, match="requires external_plan"):
        CONFIG.resolve_npu_hisparse_config(args(npu=npu), env)


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"fused_sparse_attention_mode": "reference"}, "requires.*fused"),
        ({"fsa_execution_abi": "row_v0"}, "shared_v3"),
        ({"fsa_load_mode": "COPY"}, "FUSED"),
    ],
)
def test_external_plan_rejects_unsupported_execution_settings(fields, match):
    with pytest.raises(ValueError, match=match):
        validate(args(npu={**budget(), **fields}))


def test_external_plan_initializes_with_only_the_fsa_loader(monkeypatch):
    from types import ModuleType
    from unittest.mock import Mock

    import torch

    from sglang.srt import runtime_context
    from sglang.srt.hardware_backend.npu.hisparse.fsa_backend import (
        initialize_fsa_backend,
    )
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        FusedSparseAttentionState,
    )

    for name, _ in CONFIG.ENV_FIELDS.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION", "1")
    wheel = ModuleType("sgl_kernel_npu.hisparse")
    wheel.load_fsa = Mock()
    monkeypatch.setitem(sys.modules, wheel.__name__, wheel)
    dp_attention = ModuleType("sglang.srt.layers.dp_attention")
    dp_attention.get_eager_max_batch_size = Mock(return_value=4)
    monkeypatch.setitem(sys.modules, dp_attention.__name__, dp_attention)
    monkeypatch.setattr(runtime_context, "get_disagg", lambda: args("decode"))
    monkeypatch.setattr(runtime_context, "get_spec", lambda: SimpleNamespace())
    decode = SimpleNamespace(backend="disabled", bs=None, max_bs=None)
    monkeypatch.setattr(
        runtime_context,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=decode))
        ),
    )
    pool = SimpleNamespace(size=2, req_generation=torch.zeros(3, dtype=torch.int64))
    backend = SimpleNamespace(
        page_size=128,
        qk_rope_head_dim=0,
        kv_lora_rank=512,
        tp_q_head_num=16,
        dsa_index_kpool=4,
        device="cpu",
        req_to_token_pool=pool,
        token_to_kv_pool=SimpleNamespace(start_layer=0, layer_num=1),
    )
    runner = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        model_config=SimpleNamespace(hf_text_config=model()),
    )
    monkeypatch.setenv("SGLANG_NPU_DSA_OFFLOAD_BACKEND", "lim_copy_sfa")
    with pytest.raises(ValueError, match="requires external_plan"):
        initialize_fsa_backend(backend, runner)
    wheel.load_fsa.assert_not_called()
    monkeypatch.delenv("SGLANG_NPU_DSA_OFFLOAD_BACKEND")
    initialize_fsa_backend(backend, runner)
    wheel.load_fsa.assert_called_once_with()
    dp_attention.get_eager_max_batch_size.assert_called_once_with(pool.size)
    assert isinstance(backend.fused_sparse_state, FusedSparseAttentionState)
    assert backend.fused_sparse_state.shared_topk == 2051
    assert backend.fused_sparse_state.batch_capacity == 4
    assert backend.fused_sparse_state.row_capacity == 4


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"kv_cache_dtype": "fp8_e4m3"}, "storage"),
        ({"kv_cache_dtype": "float16"}, "BF16"),
        ({"dtype": "float16"}, "BF16"),
        ({"attention_backend": "flashmla_sparse"}, "ascend"),
        ({"dsa_decode_backend": "flashmla_sparse"}, "Ascend"),
        ({"enable_unified_cache_external_linker": True}, "Host tier"),
        ({"hicache_storage_backend": "mooncake"}, "Host tier"),
        ({"enable_hierarchical_cache": False}, "HiCache"),
        ({"disable_radix_cache": True}, "radix"),
        ({"enable_mixed_chunk": True}, "CP1/DCP1"),
        ({"attn_cp_size": 2}, "CP1/DCP1"),
        ({"dcp_size": 2}, "CP1/DCP1"),
        ({"page_size": 64}, "128-token attention pages"),
        (
            {"speculative_algorithm": "EAGLE", "speculative_num_steps": 6},
            "routes<=6",
        ),
        (
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 3,
                "speculative_eagle_topk": 2,
            },
            "topk=1",
        ),
        (
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 3,
                "speculative_adaptive": True,
            },
            "fixed-width",
        ),
        (
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 3,
                "speculative_num_draft_tokens": 3,
            },
            "draft_tokens",
        ),
    ],
)
def test_existing_unsupported_service_combinations_still_fail(changes, match):
    with pytest.raises(ValueError, match=match):
        validate(args(npu=budget(), **changes))


@pytest.mark.parametrize("steps", [1, 3, 5])
def test_fixed_chain_mtp_and_larger_resident_inputs_are_accepted(steps):
    validate(
        args(
            npu={**budget(), "device_buffer_size": 8192},
            speculative_algorithm="NEXTN",
            speculative_num_steps=steps,
        )
    )


def test_external_plan_accepts_uncompressed_topk1024():
    server = args(
        npu={**budget(), "dsa_offload_backend": "external_plan"}, page_size=128
    )
    server.hisparse_config["top_k"] = 1024
    validate(server, hf=model(index_topk=1024, index_kpool=1))


@pytest.mark.parametrize("backend", ["full", "breakable"])
def test_layerwise_prefill_rejects_captured_prefill(backend):
    graph = SimpleNamespace(prefill=SimpleNamespace(backend=backend))
    with pytest.raises(ValueError, match="Prefill graph capture"):
        validate(args(npu=budget(), cuda_graph_config=graph))


def test_topk_is_model_index_units_and_not_expanded_kpool_tokens():
    server = args(npu=budget())
    server.hisparse_config["top_k"] = 2051
    with pytest.raises(ValueError, match="before KPool expansion"):
        validate(server)
    server.hisparse_config.update(top_k=2048, device_buffer_size=2048)
    with pytest.raises(ValueError, match="expanded KPool"):
        validate(server)


def test_glm53_nope_uses_effective_model_kv_geometry():
    validate(args(npu=budget()), hf=model(qk_rope_head_dim=64, use_mla_nope=True))


@pytest.mark.parametrize("is_dsa,is_v4", [(False, False), (False, True), (True, True)])
def test_npu_dsv4_and_non_dsa_are_not_blanket_waived(is_dsa, is_v4):
    with pytest.raises(ValueError, match="DeepSeek V4 is unsupported"):
        validate(args(npu=budget()), is_dsa=is_dsa, is_v4=is_v4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
