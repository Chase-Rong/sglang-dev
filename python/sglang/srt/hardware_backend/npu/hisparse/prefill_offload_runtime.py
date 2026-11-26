"""Batch ownership and real KV-pool boundaries for layerwise Prefill offload."""

import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import torch

from sglang.srt.hardware_backend.npu.hisparse.prefill_offload import (
    HostKvPagePool,
    HostPageAllocation,
    KvPageRange,
    LayerSlotToken,
    LayerWork,
    PrefillLayerSlotManager,
    SlotBusyError,
)


def validate_prefill_parallel_layout(
    *, cp_size, dcp_size, replicated_cp, layer_split, all_cp_transfer
):
    if dcp_size != 1 or layer_split:
        raise ValueError(
            "bounded NPU Prefill requires replicated KV, not DCP or layer-sharded KV"
        )
    if cp_size > 1 and not replicated_cp:
        raise ValueError(
            "bounded NPU Prefill CP requires DSA all-gather KV reconstruction"
        )
    if cp_size > 1 and all_cp_transfer:
        raise ValueError(
            "replicated bounded Prefill KV uses CP rank 0; disable all-CP transfer"
        )


@dataclass
class PrefillRequestRows:
    allocation: object
    prefix_length: int
    sequence_length: int
    token_indices: torch.Tensor
    device_rows: torch.Tensor
    source_pages: tuple
    sender: object
    prefix_proof: object = None
    mapping_borrow: object = None
    restore_mask: object = None
    cpu_device_rows: object = None
    row_plans: object = None


@dataclass
class PrefillBatchLayer:
    slot_token: LayerSlotToken
    works: list = field(default_factory=list)
    finished: bool = False
    published: bool = False
    transfers: list = field(default_factory=list)
    failed: bool = False
    publication: object = None
    restore_event: object = None


class NpuPrefillOffloadRuntime:
    """Own one physical layer slot for all requests in a Prefill batch.

    ``publish`` returns operations whose state becomes ``acknowledged`` only
    after receiver visibility is confirmed. A failed or timed-out operation
    retains all layer bindings; cancellation must explicitly drain it.
    """

    def __init__(
        self, pool, request_pool, adapter, *, publish, timeout_s=30.0, layer_offset=None
    ):
        if pool.prefill_layer_slots is None:
            raise ValueError("Prefill runtime requires a bounded physical layer pool")
        if timeout_s <= 0:
            raise ValueError("Prefill drain timeout must be positive")
        self.pool = pool
        self.request_pool = request_pool
        self.adapter = adapter
        self.publish = publish
        self.timeout_s = timeout_s
        self._transfer_layer_offset = layer_offset
        self.slots = PrefillLayerSlotManager(pool.layer_num, pool.prefill_layer_slots)
        self.backing = getattr(pool, "decode_host_backing", None)
        self.page_binding = None if self.backing is None else self.backing.page_binding
        if self.backing is not None and self.page_binding is None:
            raise ValueError("cache-backed Prefill requires allocator page binding")
        self.host_pages = (
            None
            if self.backing is not None
            else HostKvPagePool(pool.size // pool.page_size + 1, pool.page_size)
        )
        self._borrowed_allocations = {}
        self._allocation_generation = 0
        self.requests = ()
        self.layers = {}
        self.batch_id = 0
        self.active = False
        self.failed = False
        self.completed_layers = set()
        self.failure_event = None
        self._publisher = None
        self.h2d_stream = None
        self.d2h_stream = None

    def _transfer_token(self, token):
        layer_ids = getattr(self.pool, "fsa_wire_layer_ids", None)
        layer = (
            token.logical_layer if layer_ids is None else layer_ids[token.logical_layer]
        )
        offset = self._transfer_layer_offset
        if offset is None:
            offset = getattr(self.pool, "fsa_transfer_layer_offset", 0)
        return replace(token, logical_layer=layer + offset)

    def register_host_layers(self):
        if self.backing is not None:
            self.adapter.borrow_host_layers(self.backing, self.pool)
            return
        rows = self.host_pages.page_count * self.pool.page_size
        for layer_id in range(
            self.pool.start_layer, self.pool.start_layer + self.pool.layer_num
        ):
            self.adapter.register_layer_components(
                layer_id,
                latent_shape=(rows, self.pool.kv_lora_rank),
                rope_shape=(rows, self.pool.qk_rope_head_dim),
                indexer_shape=(
                    None
                    if self.pool.index_head_dim is None
                    or getattr(self.pool, "kpool_use_compress", False)
                    else (rows, self.pool.index_head_dim)
                ),
                dtype=self.pool.store_dtype,
            )

    @contextlib.contextmanager
    def forward(self, batch):
        self.begin_batch(batch)
        try:
            yield
            self.finish_batch()
        except BaseException:
            self.failed = True
            self.failure_event = self.adapter.record_event()
            for layer in self.layers.values():
                layer.failed = True
                self.slots.mark_failed(layer.slot_token)
            raise

    def begin_batch(self, batch):
        if self.active or self.layers or self.failed:
            raise SlotBusyError("previous Prefill batch has not drained")
        mode = batch.forward_mode
        if not mode.is_extend() or mode.is_target_verify() or mode.is_draft_extend_v2():
            raise ValueError("bounded Prefill runtime requires ordinary eager extend")
        request_ids = tuple(batch.rids or ())
        if not request_ids or len(set(request_ids)) != len(request_ids):
            raise ValueError("Prefill batch requires distinct request IDs")
        if batch.seq_lens_cpu is None or batch.extend_prefix_lens_cpu is None:
            raise ValueError(
                "Prefill offload requires CPU sequence and prefix metadata"
            )
        lengths = [int(value) for value in batch.seq_lens_cpu]
        prefixes = [int(value) for value in batch.extend_prefix_lens_cpu]
        request_slots = batch.req_pool_indices.detach().cpu().tolist()
        senders = batch.prefill_offload_senders
        if not (
            len(lengths)
            == len(prefixes)
            == len(request_slots)
            == len(request_ids)
            == len(senders or ())
        ):
            raise ValueError("Prefill request metadata counts must match")
        for prefix, length, slot, sender in zip(
            prefixes, lengths, request_slots, senders
        ):
            if not 0 <= prefix < length <= self.request_pool.req_to_token.shape[1]:
                raise ValueError("invalid Prefill token interval")
            if not 0 < slot < self.request_pool.req_to_token.shape[0]:
                raise ValueError("invalid Prefill request pool slot")
            if sender is None:
                raise RuntimeError("Prefill request has no layer-ready sender")
            sender.require_prefill_layer_targets(
                [
                    self._transfer_token(
                        LayerSlotToken(0, request_ids[0], layer, 0)
                    ).logical_layer
                    for layer in range(
                        self.pool.start_layer,
                        self.pool.start_layer + self.pool.layer_num,
                    )
                ]
            )
        if self.backing is not None:
            self._begin_cache_batch(
                batch, request_ids, lengths, prefixes, request_slots, senders
            )
            return
        prepared = []
        newly_allocated = []
        occupied_device_pages = set()
        try:
            for request_id, length, prefix, slot, sender in zip(
                request_ids, lengths, prefixes, request_slots, senders
            ):
                try:
                    allocation = self.host_pages.allocation(request_id)
                except KeyError:
                    if prefix:
                        raise RuntimeError(
                            "cannot restore an unowned Prefill prefix"
                        ) from None
                    allocation = self.host_pages.allocate(request_id, length)
                    newly_allocated.append(allocation)
                else:
                    allocation = self.host_pages.grow(request_id, length)
                rows = (
                    self.request_pool.req_to_token[slot, :length]
                    .to(torch.int64)
                    .clone()
                    .contiguous()
                )
                row_list = rows.detach().cpu().tolist()
                capacity = (
                    self.pool.size // self.pool.page_size + 1
                ) * self.pool.page_size
                if any(row < 0 or row >= capacity for row in row_list):
                    raise ValueError(
                        "request mapping exceeds physical KV token capacity"
                    )
                source_pages = tuple(
                    row // self.pool.page_size
                    for row in row_list[:: self.pool.page_size]
                )
                if len(set(source_pages)) != len(
                    source_pages
                ) or occupied_device_pages.intersection(source_pages):
                    raise ValueError(
                        "Prefill requests must own disjoint physical device pages"
                    )
                occupied_device_pages.update(source_pages)
                if any(
                    row
                    != source_pages[index // self.pool.page_size] * self.pool.page_size
                    + index % self.pool.page_size
                    for index, row in enumerate(row_list)
                ):
                    raise ValueError(
                        "Prefill transfer requires page-aligned request token mapping"
                    )
                prepared.append(
                    PrefillRequestRows(
                        allocation,
                        prefix,
                        length,
                        torch.arange(length, device=rows.device, dtype=torch.int64),
                        rows,
                        source_pages,
                        sender,
                    )
                )
            self._validate_batch_identity(prepared)
        except BaseException:
            for allocation in newly_allocated:
                self.host_pages.release(allocation)
            raise
        self._activate_batch(prepared)

    def _activate_batch(self, prepared):
        self.batch_id += 1
        self.requests = tuple(prepared)
        self.completed_layers.clear()
        self.active = True
        if str(self.pool.device).split(":")[0] == "npu":
            if self.h2d_stream is None:
                self.h2d_stream = torch.npu.Stream(device=self.pool.device)
                self.d2h_stream = torch.npu.Stream(device=self.pool.device)
            self.h2d_stream.wait_stream(torch.npu.current_stream())
            if (
                self.page_binding is not None
                and self.page_binding.ready_event is not None
            ):
                self.h2d_stream.wait_event(self.page_binding.ready_event)
                torch.npu.current_stream().wait_event(self.page_binding.ready_event)

    def _begin_cache_batch(self, batch, request_ids, lengths, prefixes, slots, senders):
        proofs = getattr(batch, "prefill_cache_prefix_proofs", None)
        proofs = [None] * len(request_ids) if proofs is None else proofs
        if len(proofs) != len(request_ids):
            raise ValueError("cache prefix proof count differs from the batch")
        prepared = []
        borrows = []
        allocations = dict(self._borrowed_allocations)
        try:
            for rid, length, prefix, slot, sender, proof in zip(
                request_ids, lengths, prefixes, slots, senders, proofs
            ):
                rows = (
                    self.request_pool.req_to_token[slot, :length]
                    .detach()
                    .to(torch.int64)
                    .clone()
                    .contiguous()
                )
                values = rows.cpu().tolist()
                pages = tuple(
                    row // self.pool.page_size for row in values[:: self.pool.page_size]
                )
                if len(set(pages)) != len(pages) or any(
                    row
                    != pages[i // self.pool.page_size] * self.pool.page_size
                    + i % self.pool.page_size
                    for i, row in enumerate(values)
                ):
                    raise ValueError(
                        "Prefill transfer requires distinct aligned device pages"
                    )
                if proof is not None:
                    shared = proof.shared_prefix_len
                    # One allocator page already owns a complete compressed index page.
                    if (
                        proof.shared_group_size <= 0
                        or proof.shared_group_size % self.pool.page_size
                    ):
                        raise ValueError(
                            "shared cache proof has an incomplete compressed group"
                        )
                    if (
                        shared < 0
                        or shared > prefix
                        or shared % proof.shared_group_size
                        or len(proof.token_ids) < shared
                        or tuple(values[:shared]) != proof.device_rows
                        or len(proof.device_rows) != shared
                    ):
                        raise ValueError(
                            "shared cache proof differs from admitted request rows"
                        )
                borrow = self.page_binding.borrow_mapped_pages(pages)
                borrows.append(borrow)
                previous = allocations.get(rid)
                host_pages = tuple(borrow.host_pages.tolist())
                host_generations = tuple(borrow.host_generations.tolist())
                if (
                    previous is None
                    or previous.page_ids != host_pages[: len(previous.page_ids)]
                    or previous.page_generations
                    != host_generations[: len(previous.page_ids)]
                ):
                    self._allocation_generation += 1
                    generation = self._allocation_generation
                else:
                    generation = previous.generation
                allocation = HostPageAllocation(
                    rid, host_pages, self.pool.page_size, generation, host_generations
                )
                prepared.append(
                    PrefillRequestRows(
                        allocation,
                        prefix,
                        length,
                        torch.arange(length, device=rows.device, dtype=torch.int64),
                        rows,
                        pages,
                        sender,
                        proof,
                        borrow,
                        torch.ones(length, dtype=torch.bool),
                        torch.tensor(values, dtype=torch.int64),
                    )
                )
                allocations[rid] = allocation
            self._validate_cache_aliases(prepared)
            self._build_cache_row_plans(prepared)
            self._validate_batch_identity(prepared)
        except BaseException:
            for borrow in borrows:
                self.page_binding.release_borrow(borrow)
            raise
        for request in prepared:
            previous = self._borrowed_allocations.get(request.allocation.request_id)
            if (
                previous is not None
                and previous.generation != request.allocation.generation
            ):
                self.adapter.rebind_request_coverage(
                    previous,
                    request.allocation,
                    shared_prefix_length=0
                    if request.prefix_proof is None
                    else request.prefix_proof.shared_prefix_len,
                )
        self._borrowed_allocations = allocations
        self._activate_batch(prepared)

    def _validate_cache_aliases(self, requests):
        device_uses, host_uses, restored = {}, {}, {}
        registry = {}
        size = self.pool.page_size
        for request in requests:
            proof = request.prefix_proof
            identities = None if proof is None else proof.content_identities(registry)
            for logical, (device, host) in enumerate(
                zip(request.source_pages, request.allocation.page_ids)
            ):
                offset = logical * size
                identity = None
                if proof is not None and offset + size <= min(
                    request.prefix_length, proof.shared_prefix_len
                ):
                    identity = proof.content_identity(offset, identities)
                use = (identity, device, host)
                for table, address in ((device_uses, device), (host_uses, host)):
                    old = table.get(address)
                    if old is not None and (
                        identity is None or old[0] != identity or old[2] != host
                    ):
                        raise ValueError(
                            "shared Prefill page lacks matching public cache proof or overlaps a write"
                        )
                    table[address] = use
                if offset >= request.prefix_length:
                    continue
                if device in restored:
                    if restored[device] != host:
                        raise ValueError(
                            "shared restore destination has conflicting Host sources"
                        )
                    request.restore_mask[
                        offset : min(offset + size, request.prefix_length)
                    ] = False
                else:
                    restored[device] = host

    def _build_cache_row_plans(self, requests):
        from sglang.srt.hardware_backend.npu.prefill_dma import mapped_row_runs

        size = self.pool.page_size
        device_capacity = self.page_binding.device_to_host_cpu.numel() * size
        host_capacity = self.backing.pages * size
        for request in requests:
            rows = request.cpu_device_rows
            logical = torch.arange(rows.numel(), dtype=torch.int64)
            host_pages = torch.tensor(request.allocation.page_ids, dtype=torch.int64)
            host_rows = host_pages[logical // size] * size + logical.remainder(size)
            prefix = request.prefix_length
            restore = request.restore_mask[:prefix]
            request.row_plans = (
                mapped_row_runs(
                    host_rows[:prefix][restore],
                    rows[:prefix][restore],
                    host_capacity,
                    device_capacity,
                ),
                mapped_row_runs(
                    rows[prefix:],
                    host_rows[prefix:],
                    device_capacity,
                    host_capacity,
                ),
            )

    def _release_batch_borrows(self):
        if self.page_binding is not None:
            for request in self.requests:
                self.page_binding.release_borrow(request.mapping_borrow)

    def _validate_batch_identity(self, requests):
        """Validate shared ownership before rank-local asynchronous prefetch."""
        shared_group = getattr(self.adapter.host_pool, "shared_group", None)
        if shared_group is None:
            return
        shared_group.validate_identity(
            (
                "prefill_batch",
                self.batch_id + 1,
                self.pool.start_layer,
                self.pool.layer_num,
                self.pool.prefill_layer_slots,
                self.pool.page_size,
                tuple(
                    (
                        request.allocation.request_id,
                        request.allocation.generation,
                        tuple(request.allocation.page_ids),
                        request.prefix_length,
                        request.sequence_length,
                        request.source_pages,
                    )
                    for request in requests
                ),
            )
        )

    def prepare_layer(self, layer_id):
        self._prepare_layer(layer_id)
        layer = self.layers[layer_id]
        if layer.restore_event is not None:
            torch.npu.current_stream().wait_event(layer.restore_event)
        self._prefetch_next_layer(layer_id)

    def _prefetch_next_layer(self, layer_id):
        next_layer = layer_id + 1
        if (
            self.h2d_stream is None
            or self.pool.prefill_layer_slots < 2
            or next_layer >= self.pool.start_layer + self.pool.layer_num
            or next_layer in self.layers
            or next_layer in self.completed_layers
        ):
            return
        slot_id = self.slots.preferred_slot(next_layer - self.pool.start_layer)
        if self.slots.owner(slot_id) is not None:
            previous_id, previous = next(
                (layer, state)
                for layer, state in self.layers.items()
                if state.slot_token.slot_id == slot_id
            )
            if (
                not previous.finished
                or previous.failed
                or previous.publication is not None
                and not previous.publication.done()
            ):
                return
            self._finish_publication(previous, time.monotonic() + self.timeout_s)
            if any(
                operation.state != "acknowledged" for operation in previous.transfers
            ):
                return
            try:
                for work in previous.works:
                    self.adapter.validate_layer_release(work)
            except SlotBusyError:
                return
            self.drain_layer(previous_id)
        with torch.npu.stream(self.h2d_stream):
            self._prepare_layer(next_layer)
            event = torch.npu.Event()
            event.record(self.h2d_stream)
            self.layers[next_layer].restore_event = event

    def _prepare_layer(self, layer_id):
        if not self.active or self.failed:
            raise RuntimeError("KV access has no active healthy Prefill batch")
        if layer_id in self.completed_layers:
            raise RuntimeError(
                "Prefill layer was accessed after its batch work completed"
            )
        if layer_id in self.layers:
            if self.layers[layer_id].finished:
                raise RuntimeError(
                    "Prefill layer was accessed after attention completion"
                )
            return
        local_layer = layer_id - self.pool.start_layer
        slot_id = self.slots.preferred_slot(local_layer)
        for previous_id, previous in tuple(self.layers.items()):
            if previous.slot_token.slot_id == slot_id:
                self.drain_layer(previous_id)
        token = self.slots.acquire(f"batch:{self.batch_id}", local_layer)
        layer = PrefillBatchLayer(token)
        self.layers[layer_id] = layer
        tensors = self.pool.raw_layer_components(layer_id)
        if getattr(self.pool, "kpool_use_compress", False):
            tensors = (tensors[0], tensors[1], None)
        try:
            for request in self.requests:
                work = LayerWork(
                    LayerSlotToken(
                        slot_id,
                        request.allocation.request_id,
                        layer_id,
                        token.generation,
                    ),
                    request.allocation,
                    KvPageRange(request.prefix_length, request.sequence_length),
                )
                self.adapter.bind_layer(
                    work,
                    *tensors,
                    token_indices=request.token_indices,
                    device_row_indices=request.device_rows,
                    source_page_ids=request.source_pages,
                    cache_prefix_length=(
                        0
                        if request.prefix_proof is None
                        else request.prefix_proof.shared_prefix_len
                    ),
                    restore_mask=request.restore_mask,
                    row_plans=request.row_plans,
                )
                layer.works.append(work)
                self.adapter.restore_layer(work)
                work.restored = True
        except BaseException:
            layer.failed = True
            self.failed = True
            self.slots.mark_failed(token)
            raise

    def finish_layer(self, layer_id):
        layer = self.layers[layer_id]
        if layer.finished or layer.failed:
            raise RuntimeError("Prefill layer cannot complete twice or after failure")
        for work in layer.works:
            work.kv_written = True
            self.adapter.record_attention_done(work)
        if self._publisher is None:
            self._publisher = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="npu-pd-staging"
            )
        layer.published = True
        layer.publication = self._publisher.submit(
            self._publish_layer, layer, tuple(self.requests)
        )
        context = contextlib.nullcontext()
        if self.d2h_stream is not None:
            for work in layer.works:
                self.d2h_stream.wait_event(self.adapter.attention_ready_event(work))
            context = torch.npu.stream(self.d2h_stream)
        with context:
            for work in layer.works:
                work.host_save_started = True
                self.adapter.save_layer(work)
        layer.finished = True

    def _publish_layer(self, layer, requests):
        if str(self.pool.device).split(":")[0] == "npu":
            torch.npu.set_device(self.pool.device)
        deadline = time.monotonic() + self.timeout_s
        for request, work in zip(requests, layer.works):
            event = self.adapter.attention_ready_event(work)
            self._wait(event.query, deadline, "Prefill KV-ready event timed out")
            try:
                layer.transfers.extend(
                    self.publish(
                        request.sender, work, self.adapter.binding(work), event
                    )
                )
            except BaseException:
                # A publisher can register one receiver operation and then fail
                # before returning. Keep that operation and its source lease.
                token = self._transfer_token(work.token)
                known = {id(operation) for operation in layer.transfers}
                for operation in request.sender.prefill_layer_operations():
                    if operation.record.token == token and id(operation) not in known:
                        layer.transfers.append(operation)
                        known.add(id(operation))
                raise

    def _finish_publication(self, layer, deadline):
        if layer.publication is not None:
            layer.publication.result(timeout=max(0, deadline - time.monotonic()))
            layer.publication = None
            self.slots.set_pd_readers(layer.slot_token, len(layer.transfers))

    def _wait(self, predicate, deadline, reason):
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(reason)
            time.sleep(0.001)

    def drain_layer(self, layer_id):
        layer = self.layers[layer_id]
        if not layer.finished or layer.failed:
            raise SlotBusyError(
                "cannot reuse a layer before successful attention completion"
            )
        deadline = time.monotonic() + self.timeout_s
        self._finish_publication(layer, deadline)
        for work in layer.works:
            self.adapter.wait_local_completion(work, deadline=deadline)
            self.adapter.complete_host_save(work)
            self.adapter.complete_attention(work)
        self.slots.mark_host_saved(layer.slot_token)
        self.slots.mark_attention_done(layer.slot_token)

        def acknowledged():
            if any(
                operation.state in ("failed", "drained")
                for operation in layer.transfers
            ):
                layer.failed = True
                self.slots.mark_failed(layer.slot_token)
                raise RuntimeError("Prefill layer transfer failed")
            return all(
                operation.state == "acknowledged" for operation in layer.transfers
            )

        self._wait(acknowledged, deadline, "Prefill layer receiver ACK timed out")
        for _ in layer.transfers:
            self.slots.mark_pd_read_done(layer.slot_token)
        for work in layer.works:
            self.adapter.validate_layer_release(work)
        for work in layer.works:
            self.adapter.release_layer(work)
        self.slots.release(layer.slot_token)
        del self.layers[layer_id]
        self.completed_layers.add(layer_id)

    def finish_batch(self):
        for layer_id in tuple(self.layers):
            self.drain_layer(layer_id)
        if len(self.completed_layers) != self.pool.layer_num:
            raise RuntimeError(
                "Prefill batch did not execute every local attention layer"
            )
        if self.backing is not None:
            shared = getattr(self.backing.pool, "shared_group", None)
            if shared is not None:
                shared.publish_write()
            # All local saves and remote readers have drained. This fresh event
            # belongs to this batch, never to an earlier cache checkpoint.
            self.page_binding.ready_event = self.adapter.record_event()
            self.adapter.commit_host_saves()
            self._release_batch_borrows()
        self.requests = ()
        self.active = False

    def can_release_request(self, request_id):
        return not (
            any(
                request.allocation.request_id == request_id for request in self.requests
            )
            or any(
                work.token.request_id == request_id
                for layer in self.layers.values()
                for work in layer.works
            )
        )

    def drain_for_shutdown(self):
        if self.active or self.layers or self.failed:
            self.failed = True
            if self.failure_event is None:
                self.failure_event = self.adapter.record_event()
            for request in self.requests:
                request.sender.abort()
            deadline = time.monotonic() + self.timeout_s
            for layer in self.layers.values():
                if layer.publication is not None:
                    self._wait(
                        layer.publication.done,
                        deadline,
                        "Prefill publication has not drained during shutdown",
                    )
            operations = [
                operation
                for layer in self.layers.values()
                for operation in layer.transfers
            ]
            for request in self.requests:
                operations.extend(request.sender.prefill_layer_operations())
            self._wait(
                lambda: all(
                    operation.state in ("acknowledged", "drained")
                    or getattr(operation, "transport_drained", False)
                    for operation in operations
                ),
                deadline,
                "Prefill transport has not drained during shutdown",
            )
            for operation in operations:
                if operation.state not in ("acknowledged", "drained"):
                    operation.state = "drained"
            self.drain_failed_batch()
        if self._publisher is not None:
            self._publisher.shutdown(wait=True)
            self._publisher = None
        for stream in (self.h2d_stream, self.d2h_stream):
            if stream is not None:
                stream.synchronize()

    def release_request(self, request_id):
        if not self.can_release_request(request_id):
            raise SlotBusyError("request release requires its Prefill work to drain")
        if self.backing is not None:
            allocation = self._borrowed_allocations.pop(request_id, None)
            if allocation is not None:
                self.adapter.release_request(allocation)
            return
        try:
            allocation = self.host_pages.allocation(request_id)
        except KeyError:
            return
        self.adapter.release_request(allocation)
        self.host_pages.release(allocation)

    def drain_failed_batch(self):
        if not self.failed or self.failure_event is None:
            raise RuntimeError("failed-batch drain requires a recorded failure event")
        deadline = time.monotonic() + self.timeout_s
        self._wait(
            self.failure_event.query,
            deadline,
            "failed Prefill device work has not drained",
        )
        for layer in self.layers.values():
            if layer.publication is not None:
                if not layer.publication.done():
                    raise SlotBusyError("failed Prefill publication has not drained")
                layer.publication.exception()
                layer.publication = None
                self.slots.set_pd_readers(layer.slot_token, len(layer.transfers))
        tokens = {
            self._transfer_token(work.token)
            for layer in self.layers.values()
            for work in layer.works
        }
        for layer in self.layers.values():
            if any(
                operation.state not in ("acknowledged", "drained")
                for operation in layer.transfers
            ):
                raise SlotBusyError("failed Prefill transfer has not drained")
        for request in self.requests:
            for operation in request.sender.prefill_layer_operations():
                if operation.record.token in tokens and operation.state not in (
                    "acknowledged",
                    "drained",
                ):
                    raise SlotBusyError("failed Prefill transfer has not drained")
        for layer in self.layers.values():
            for work in layer.works:
                self.adapter.wait_local_completion(work, deadline=deadline)
                self.adapter.release_layer(work)
            self.slots.mark_failed(layer.slot_token)
            self.slots.mark_failed_local_work_drained(layer.slot_token)
            readers = self.slots.pending_pd_readers(layer.slot_token)
            if readers:
                self.slots.mark_pd_read_done(layer.slot_token, readers)
            self.slots.release(layer.slot_token)
        self.layers.clear()
        self.adapter.discard_host_saves()
        self._release_batch_borrows()
        self.requests = ()
        self.active = False
        self.failed = False
        self.failure_event = None


class PrefillReleaseCheck:
    def __init__(self, runtimes, request_id, sender):
        self.runtimes = runtimes
        self.request_id = request_id
        self.sender = sender

    def poll(self):
        from sglang.srt.disaggregation.base.conn import KVPoll

        manager = getattr(self.sender, "kv_mgr", None)
        room = getattr(self.sender, "bootstrap_room", None)
        if manager is not None and (
            getattr(manager, "_staging_outstanding", {}).get(room, 0) > 0
            or room in getattr(manager, "_layer_poisoned_rooms", ())
            or any(
                identity[0] == room
                for identity in getattr(manager, "_layer_pending_aborts", ())
            )
        ):
            return KVPoll.WaitingForInput
        for runtime in self.runtimes:
            if getattr(runtime, "failed", False):
                try:
                    runtime.drain_failed_batch()
                except (SlotBusyError, TimeoutError):
                    return KVPoll.WaitingForInput
        if any(
            not runtime.can_release_request(self.request_id)
            for runtime in self.runtimes
        ):
            return KVPoll.WaitingForInput
        if any(
            operation.state not in ("acknowledged", "drained")
            for operation in self.sender.prefill_layer_operations()
        ):
            return KVPoll.WaitingForInput
        return KVPoll.Success
