"""Initialize the bounded Prefill runtime for integrated HiSparse."""

from sglang.srt.environ import envs


def initialize_prefill_runtime(runner):
    if not envs.SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD.get():
        return
    from sglang.srt.hardware_backend.npu.hisparse.prefill_overlap import (
        PipelinedIntegratedOffloadRuntime,
    )

    pool = getattr(runner.token_to_kv_pool, "full_kv_pool", runner.token_to_kv_pool)
    runtime = PipelinedIntegratedOffloadRuntime(pool, runner.req_to_token_pool)
    runner.prefill_offload_runtime = runtime
    runner.token_to_kv_pool.prefill_offload_runtime = runtime
