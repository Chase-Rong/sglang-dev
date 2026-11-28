"""Fused attention adapter for the baseline's NPU DSA and KPool metadata."""

from dataclasses import replace

from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
    FusedSparseAttentionState,
    paged_kv_view,
    run_fused_sparse_attention,
)
from sglang.srt.hardware_backend.npu.hisparse.hicache_offload import fsa_resident_epoch


def sparse_pool_and_layer(pool, layer_id):
    if hasattr(pool, "full_kv_pool"):
        layer_id = pool._transfer_full_attention_id(layer_id)
        pool = pool.full_kv_pool
    return pool, layer_id - pool.start_layer


def sparse_layer_ids(pool):
    if hasattr(pool, "full_kv_pool"):
        return tuple(pool.full_attention_layer_id_mapping)
    return tuple(range(pool.start_layer, pool.start_layer + pool.layer_num))


def initialize_fsa_backend(backend, model_runner):
    from sglang.srt.environ import envs

    from .config import validate_dsa_offload_backend

    backend.fused_sparse_state = None
    validate_dsa_offload_backend(envs.SGLANG_NPU_DSA_OFFLOAD_BACKEND.get())
    if not envs.SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION.get():
        return
    from sglang.srt.hardware_backend.npu.fsa_native import (
        SHARED_QUERY_HEADS,
        native_ops,
    )
    from sglang.srt.hardware_backend.npu.hisparse.speculative_sparse import (
        sparse_mtp_rows,
        sparse_mtp_tail,
    )
    from sglang.srt.layers.dp_attention import get_eager_max_batch_size
    from sglang.srt.runtime_context import get_exec, get_spec

    from .config import (
        register_shared_execution,
        register_shared_geometry,
        resolve_fsa_execution_abi,
        resolve_shared_load_mode,
        resolve_shared_warmup_threshold,
        shared_union_width,
    )

    execution_abi = resolve_fsa_execution_abi()
    load_mode = resolve_shared_load_mode()
    warmup_threshold = resolve_shared_warmup_threshold()
    if warmup_threshold:
        from sglang.srt.runtime_context import get_parallel

        if (
            get_parallel().pp_size != 1
            or not model_runner.spec_algorithm.is_none()
            or getattr(model_runner, "is_draft_worker", False)
        ):
            raise ValueError(
                "automatic FSA warmup supports PP=1 target ordinary Decode without speculative/draft workers"
            )
    if envs.SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE.get() != "fused":
        raise ValueError("external_plan requires fused_sparse_attention_mode=fused")
    if (
        backend.page_size != 128
        or backend.qk_rope_head_dim != 0
        or backend.kv_lora_rank != 512
        or str(model_runner.dtype) != "torch.bfloat16"
        or str(model_runner.kv_cache_dtype) != "torch.bfloat16"
        or getattr(backend, "tp_q_head_num", None) not in SHARED_QUERY_HEADS
    ):
        raise ValueError("external_plan requires BF16/H16 or H32/NoPE512/RoPE0/page128")
    native_ops()
    register_shared_execution(backend.req_to_token_pool, load_mode, warmup_threshold)
    config = model_runner.model_config.hf_text_config
    from .external_plan_topology import supports_draft_topology

    draft_topology_reuse = getattr(
        model_runner, "is_draft_worker", False
    ) and supports_draft_topology(config)
    backend.fused_sparse_topk = config.index_topk + backend.dsa_index_kpool - 1
    configured_capacity = envs.SGLANG_NPU_FSA_RESIDENT_CAPACITY.get()
    routes = sparse_mtp_rows(
        model_runner.spec_algorithm, get_spec(), model_runner.model_config
    )
    from sglang.srt.hardware_backend.npu.hisparse.external_sparse_planner import (
        expanded_resident_capacity,
    )

    capacity = expanded_resident_capacity(
        configured_capacity,
        routes,
        backend.fused_sparse_topk,
        backend.page_size,
    )
    shared_union_width(routes, backend.fused_sparse_topk)
    layer_ids = sparse_layer_ids(backend.token_to_kv_pool)
    owner = backend.token_to_kv_pool
    state = getattr(owner, "fsa_runtime_state", None)
    if state is None:
        graph = get_exec().graph.cuda_graph_config.decode
        graph_requests = (
            max([graph.max_bs or 0, *(graph.bs or ())])
            if graph.backend != "disabled"
            else 0
        )
        batch_capacity = max(
            get_eager_max_batch_size(backend.req_to_token_pool.size), graph_requests
        )
        row_capacity = batch_capacity * routes
        graph_reservation = (
            graph.backend,
            tuple(graph.bs or ()),
            graph.max_bs or 0,
        )
        register_shared_geometry(
            backend.req_to_token_pool,
            topk=backend.fused_sparse_topk,
            routes=routes,
            capacity=capacity,
            page_size=backend.page_size,
            dtype=model_runner.kv_cache_dtype,
            qcap=backend.req_to_token_pool.req_generation.numel(),
            bcap=batch_capacity,
            tcap=row_capacity,
            graph_reservation=graph_reservation,
            abi=execution_abi,
        )
        backend.req_to_token_pool._shared_fsa_graph_reservation = graph_reservation
        state = FusedSparseAttentionState(
            backend.device,
            backend.req_to_token_pool,
            resident_capacity=capacity,
            layer_ids=layer_ids,
            row_capacity=row_capacity,
            mutable_tail_tokens=sparse_mtp_tail(
                model_runner.spec_algorithm, get_spec(), model_runner.model_config
            ),
            max_query_routes=routes,
            page_size=backend.page_size,
            execution_abi=execution_abi,
            batch_capacity=batch_capacity,
            load_mode=load_mode,
            warmup_threshold=warmup_threshold,
            draft_topology_reuse=draft_topology_reuse,
        )
        state.shared_topk = backend.fused_sparse_topk
        owner.fsa_runtime_state = state
    elif state.pool is not backend.req_to_token_pool:
        raise ValueError("FSA state belongs to a different request pool")
    if state.execution_abi != execution_abi:
        raise ValueError("FSA execution ABI cannot change after state allocation")
    if state.three_stage_enabled:
        from .current_kv_writeback import (
            CurrentKvWriteback,
            supports_current_kv_overlap,
        )

        if supports_current_kv_overlap():
            pool = getattr(owner, "full_kv_pool", owner)
            backing = getattr(pool, "decode_host_backing", None)
            if (
                backing is not None
                and getattr(backing, "current_writeback", None) is None
            ):
                backing.current_writeback = CurrentKvWriteback(
                    backend.device, backing.page_binding
                )
    backend.fused_sparse_state = state


def reserve_fsa_graph(backend, max_num_tokens):
    state = backend.fused_sparse_state
    if state is None:
        return
    for layer_id in sparse_layer_ids(backend.token_to_kv_pool):
        key, rope = backend.token_to_kv_pool.get_kv_buffer(layer_id)
        state.reserve(
            layer_id,
            max_num_tokens,
            backend.fused_sparse_topk,
            paged_kv_view(rope, backend.page_size, backend.qk_rope_head_dim),
            paged_kv_view(key, backend.page_size, backend.kv_lora_rank),
        )


def forward_fsa(
    backend, q, k, v, layer, forward_batch, save_kv_cache, q_rope, k_rope, topk_indices
):
    from contextlib import nullcontext

    pool, local_layer = sparse_pool_and_layer(backend.token_to_kv_pool, layer.layer_id)
    backing = getattr(pool, "decode_host_backing", None)
    writeback = getattr(backing, "current_writeback", None)
    overlap = (
        save_kv_cache
        and getattr(backend.fused_sparse_state, "three_stage_enabled", False)
        and writeback is not None
    )

    def write():
        backend.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, k_rope
        )

    if save_kv_cache and not overlap:
        write()
    if overlap:
        writeback.binding = getattr(backing, "page_binding", None)
    context = writeback.submit(write) if overlap else nullcontext()
    with context as ticket:
        return _forward_fsa(
            backend,
            q,
            layer,
            forward_batch,
            q_rope,
            topk_indices,
            backing,
            local_layer,
            ticket,
        )


def _forward_fsa(
    backend,
    q,
    layer,
    forward_batch,
    q_rope,
    topk_indices,
    backing,
    local_layer,
    current_kv_writeback,
):
    full_kv, full_rope = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    metadata = backend.forward_metadata
    lengths = metadata.cache_lens_int32
    if topk_indices is None:
        raise ValueError("FSA requires the baseline Indexer's expanded token IDs")
    if topk_indices.shape[-1] != backend.fused_sparse_topk:
        raise ValueError("FSA TopK width must include every KPool tail column")
    full_table = metadata.real_page_table
    if backing is not None:
        full_table = metadata.host_page_table
        if full_table is None:
            raise RuntimeError("Integrated FSA requires the HiCache Host page table")
    query_ends = metadata.fsa_query_ends_int32
    if query_ends is None:
        raise RuntimeError("FSA query layout must be initialized with DSA metadata")
    if q_rope is None:
        q_rope = q.new_empty((*q.shape[:-1], 0))
    mapping_epoch = 0
    if backing is not None and backing.page_binding is not None:
        mapping_epoch = fsa_resident_epoch(backing.page_binding)
    inputs = backend.fused_sparse_state.build_inputs(
        layer_id=layer.layer_id,
        q_nope=q,
        q_rope=q_rope,
        topk_indices=topk_indices,
        full_kv=paged_kv_view(full_kv, backend.page_size, backend.kv_lora_rank),
        full_rope=paged_kv_view(full_rope, backend.page_size, backend.qk_rope_head_dim),
        block_table=full_table,
        kv_lengths=lengths,
        query_lengths=query_ends,
        request_slots=forward_batch.req_pool_indices[: query_ends.numel()],
        scale=layer.scaling,
        query_positions=getattr(forward_batch, "positions", None),
        mapping_epoch=mapping_epoch,
    )
    if backing is not None:
        latent_region, rope_region = backing.regions[local_layer]
        inputs = replace(
            inputs,
            full_kv_region=latent_region,
            full_rope_region=rope_region,
            host_binding=backing.page_binding,
            current_kv_writeback=current_kv_writeback,
        )
    return run_fused_sparse_attention(inputs)
