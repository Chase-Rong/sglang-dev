import contextlib
import sys
from threading import Event as ThreadEvent
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu import prefill_dma
from sglang.srt.hardware_backend.npu.hisparse import prefill_overlap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Stream:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.tail = frozenset()

    def mark(self, name):
        self.tail = self.tail | {name}

    def wait_event(self, event):
        assert event.recorded
        self.tail = self.tail | event.dependencies

    def synchronize(self):
        self.model.synchronized.update(self.tail)


class Event:
    def __init__(self, model):
        self.model = model
        self.recorded = False

    def record(self, stream):
        self.recorded = True
        self.dependencies = stream.tail

    def query(self):
        return self.dependencies <= self.model.synchronized

    def synchronize(self):
        self.model.synchronized.update(self.dependencies)


class Device:
    def __init__(self):
        self.synchronized = set()
        self.capturing = False
        self.current = Stream(self, "compute")
        self.compute = self.current
        self.created = []

    def Stream(self, device):
        stream = Stream(self, f"copy{len(self.created)}")
        self.created.append(stream)
        return stream

    def Event(self):
        return Event(self)

    def current_stream(self):
        return self.current

    def is_current_stream_capturing(self):
        return self.capturing

    @contextlib.contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous


@pytest.fixture
def setup(monkeypatch):
    device = Device()
    monkeypatch.setattr(torch, "npu", device, raising=False)

    def build(h2d=True, d2h=True, slots=2):
        host = torch.zeros(4, 24, 2)
        storage = torch.zeros(slots, 24, 2)
        copies = []
        publications = []
        plans = []
        device_to_host = torch.tensor([0, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6])

        def host_rows(rows):
            return device_to_host[rows // 2] * 2 + rows.remainder(2)

        def plan_prefill_rows(prefix_rows, new_rows):
            prefix_rows = torch.unique(prefix_rows)
            result = (
                prefill_dma.mapped_row_runs(
                    host_rows(prefix_rows), prefix_rows, 24, 24
                ),
                prefill_dma.mapped_row_runs(new_rows, host_rows(new_rows), 24, 24),
            )
            plans.append(result)
            return result

        def restore(name, target, plan):
            stream = device.current
            copies.append(
                ("restore", name, runtime.generation, stream.name, stream.tail)
            )
            for source, destination, count in plan.tolist():
                target[destination : destination + count] = host[
                    name, source : source + count
                ]
            stream.mark(f"restore-{runtime.generation}-{name}")

        def save(layer, components, rows, valid, *, plan):
            stream = device.current
            assert f"scatter-{runtime.generation}-{layer}" in stream.tail
            copies.append(("save", layer, runtime.generation, stream.name, stream.tail))
            for source, destination, count in plan.tolist():
                host[layer, destination : destination + count] = components[0].view(
                    -1, 2
                )[source : source + count]
            stream.mark(f"save-{runtime.generation}-{layer}")

        def publish():
            publications.append(device.current.tail)

        backing = SimpleNamespace(
            pages=12,
            page_size=2,
            _component_names=[(layer, None) for layer in range(4)],
            pool=SimpleNamespace(copy_prefill_from_host=restore),
            page_binding=SimpleNamespace(
                ready_event=None,
                host_rows=host_rows,
                mapping_epoch=17,
                lock=Lock(),
                device_to_host_cpu=device_to_host,
                host_generations=torch.ones_like(device_to_host),
                check_reader_health=lambda: None,
                _validate_device_pages=lambda pages: None,
            ),
            plan_prefill_rows=plan_prefill_rows,
            enqueue_prefill_rows_from_slot=save,
            publish_prefill_writes=publish,
        )

        def physical(layer):
            if not 7 <= layer < 11:
                raise IndexError(layer)
            return (layer - 7) % slots

        pool = SimpleNamespace(
            integrated_offload=True,
            decode_host_backing=backing,
            prefill_layer_slots=slots,
            start_layer=7,
            layer_num=4,
            kv_lora_rank=2,
            qk_rope_head_dim=0,
            k_buffer=storage,
            physical_layer_index=physical,
            raw_layer_components=lambda layer: (
                storage[physical(layer)],
                torch.empty(0),
            ),
        )
        request_pool = SimpleNamespace(
            req_to_token=torch.tensor([[8, 3, 10, 5, 12, 7]]),
            req_generation=torch.tensor([41]),
        )
        runtime = prefill_overlap.PipelinedIntegratedOffloadRuntime(
            pool, request_pool, h2d_prefetch=h2d, d2h_async=d2h
        )
        return SimpleNamespace(
            runtime=runtime,
            pool=pool,
            host=host,
            copies=copies,
            publications=publications,
            device=device,
            mapping=request_pool.req_to_token[0],
            host_rows=host_rows,
            plans=plans,
        )

    return build


@pytest.mark.parametrize(
    "h2d,d2h", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("slots", [1, 2])
def test_prefix_and_slot_reuse_dependencies(setup, h2d, d2h, slots):
    case = setup(h2d, d2h, slots)
    runtime = case.runtime
    for prefix, length in ((0, 2), (2, 4), (4, 6)):
        batch = SimpleNamespace(
            extend_prefix_lens_cpu=[prefix],
            seq_lens_cpu=torch.tensor([length]),
            req_pool_indices=torch.tensor([0]),
            out_cache_loc=case.mapping[prefix:length],
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        )
        with runtime.forward(batch):
            for layer in range(7, 11):
                runtime.prepare_layer(layer)
                runtime.prepare_layer(layer)
                current = case.pool.raw_layer_components(layer)[0]
                torch.testing.assert_close(
                    current[case.mapping[:prefix]],
                    case.host[layer - 7, case.host_rows(case.mapping[:prefix])],
                )
                current[case.mapping[prefix:length]] = layer * 10 + prefix
                case.device.current.mark(f"scatter-{runtime.generation}-{layer - 7}")
                if d2h:
                    runtime.on_kv_written(layer, case.mapping[prefix:length])
                else:
                    case.host[
                        layer - 7, case.host_rows(case.mapping[prefix:length])
                    ] = current[case.mapping[prefix:length]]
                runtime.on_compute_start(layer)
                case.device.current.mark(f"attention-{runtime.generation}-{layer - 7}")
                runtime.finish_layer(layer)
        assert (
            case.pool.decode_host_backing.page_binding.ready_event
            is runtime.host_visible
        )
        assert len(case.plans) == runtime.generation
        assert all(plan.shape[1] == 3 for plan in case.plans[-1])
        assert all(torch.all(plan[:, 0] != plan[:, 1]) for plan in case.plans[-1])
        if d2h:
            assert all(
                f"save-{runtime.generation}-{layer}" in case.publications[-1]
                for layer in range(4)
            )
    for operation, layer, generation, stream, dependencies in case.copies:
        if operation == "restore" and layer >= slots:
            assert f"attention-{generation}-{layer - slots}" in dependencies
            if d2h:
                assert f"save-{generation}-{layer - slots}" in dependencies
        if operation == "restore" and generation > 1:
            assert f"attention-{generation - 1}-3" in dependencies
            if d2h:
                assert all(
                    f"save-{generation - 1}-{previous}" in dependencies
                    for previous in range(4)
                )
    runtime.drain()
    assert not runtime.retired and not runtime.active and not runtime.failed


@pytest.mark.parametrize("same_rows", [True, False])
def test_eager_buffer_copy_reuses_only_the_planned_rows(setup, same_rows):
    case = setup()
    runtime = case.runtime
    eager_locations = torch.empty(2, dtype=torch.int64)
    for prefix, length in ((0, 2), (2, 4)):
        batch = SimpleNamespace(
            extend_prefix_lens_cpu=[prefix],
            seq_lens_cpu=torch.tensor([length]),
            req_pool_indices=torch.tensor([0]),
            out_cache_loc=case.mapping[prefix:length],
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        )
        source = batch.out_cache_loc if same_rows else case.mapping[4:6]
        with runtime.forward(batch):
            eager_locations.copy_(source)
            assert source.data_ptr() != eager_locations.data_ptr()
            runtime.bind_copied_write_locations(source, eager_locations)
            for layer in range(7, 11):
                runtime.prepare_layer(layer)
                slot = case.pool.raw_layer_components(layer)[0]
                slot[eager_locations] = layer * 10 + prefix
                case.device.compute.mark(f"scatter-{runtime.generation}-{layer - 7}")
                runtime.on_kv_written(layer, eager_locations)
                if same_rows:
                    assert runtime.layers[layer]["plan"] is runtime.new_plan
                runtime.on_compute_start(layer)
                runtime.finish_layer(layer)
                torch.testing.assert_close(
                    case.host[layer - 7, case.host_rows(source)], slot[source]
                )
        assert len(case.plans) == runtime.generation * (1 if same_rows else 5)


@pytest.fixture
def queued_chunk(setup):
    case = setup()
    runtime = case.runtime
    runtime.request_pool.req_generation = torch.tensor([41, 4])
    runtime.request_pool.req_to_token = torch.tensor(
        [[8, 9, 4, 5, 16], [8, 9, 10, 11, 0]]
    )
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[3, 2],
        seq_lens_cpu=torch.tensor([5, 4]),
        req_pool_indices=torch.tensor([0, 1]),
        req_pool_indices_cpu=torch.tensor([0, 1]),
        out_cache_loc=torch.tensor([5, 16, 10, 11]),
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    planner = runtime.chunk_planner

    def submit():
        planner.submit(
            torch.tensor([4, 2, 8, 4, 5]),
            batch.out_cache_loc,
            batch.req_pool_indices,
            batch.req_pool_indices_cpu,
            batch.extend_prefix_lens_cpu,
            batch.seq_lens_cpu,
        )

    try:
        yield case, batch, submit
    finally:
        planner.close()


def test_chunk_plan_ready_before_publication_and_reused_by_eager(queued_chunk):
    case, batch, submit = queued_chunk
    runtime = case.runtime
    backing = case.pool.decode_host_backing
    expected = backing.plan_prefill_rows(
        torch.tensor([8, 9, 4, 8, 9]), batch.out_cache_loc
    )
    case.plans.clear()
    case.device.compute.mark("previous-publication")
    previous_publication = runtime._record(case.device.compute)
    runtime.host_visible = previous_publication
    submit()
    runtime.chunk_planner._inflight.result(timeout=5)
    assert not previous_publication.query()

    with runtime.forward(batch):
        torch.testing.assert_close(runtime.prefix_plan, expected[0])
        torch.testing.assert_close(runtime.new_plan, expected[1])
        copied = batch.out_cache_loc.clone()
        runtime.bind_copied_write_locations(batch.out_cache_loc, copied)
        for layer in range(7, 11):
            runtime.prepare_layer(layer)
            case.pool.raw_layer_components(layer)[0][copied] = layer
            case.device.compute.mark(f"scatter-1-{layer - 7}")
            runtime.on_kv_written(layer, copied)
            assert runtime.layers[layer]["plan"] is runtime.new_plan
            runtime.on_compute_start(layer)
            runtime.finish_layer(layer)
    assert not case.plans
    assert all(
        "previous-publication" in dependencies for *_, dependencies in case.copies
    )
    assert runtime.chunk_planner.take(batch, batch.req_pool_indices_cpu) is None


@pytest.mark.parametrize(
    "changed", ["layout", "order", "request_reuse", "host_reuse", "unrelated_mapping"]
)
def test_only_matching_admitted_layout_can_take_plan(queued_chunk, changed):
    case, batch, submit = queued_chunk
    planner = case.runtime.chunk_planner
    binding = case.pool.decode_host_backing.page_binding
    submit()
    if changed == "layout":
        batch.out_cache_loc = batch.out_cache_loc.clone()
    elif changed == "order":
        batch.req_pool_indices_cpu = batch.req_pool_indices_cpu.flip(0)
    elif changed == "request_reuse":
        case.runtime.request_pool.req_generation[0] += 1
    elif changed == "host_reuse":
        binding.host_generations[binding.device_to_host_cpu[4]] += 1
    elif changed == "unrelated_mapping":
        binding.mapping_epoch += 1
        binding.device_to_host_cpu[11] = 2
    plans = planner.take(batch, batch.req_pool_indices_cpu)
    assert (plans is not None) == (changed == "unrelated_mapping")


def test_page_spans_match_row_plans_for_fragmented_partial_chunks(queued_chunk):
    case, _, _ = queued_chunk
    runtime = case.runtime
    module = sys.modules[runtime.chunk_planner.__class__.__module__]
    generator = torch.Generator().manual_seed(1735)
    page_size = 4
    page_count = 16
    capacity = (page_count + 1) * page_size
    for _ in range(32):
        host_map = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64),
                torch.randperm(page_count, generator=generator) + 1,
            )
        )
        pages = torch.randperm(page_count, generator=generator)[:6] + 1
        seq_lens = tuple(torch.randint(9, 13, (2,), generator=generator).tolist())
        prefix_lens = tuple(
            int(torch.randint(0, length + 1, (), generator=generator))
            for length in seq_lens
        )
        per_request = [
            (request_pages[:, None] * page_size + torch.arange(page_size)).flatten()[
                :length
            ]
            for request_pages, length in zip(pages.split(3), seq_lens)
        ]
        prefix_rows = torch.cat(
            [rows[:prefix] for rows, prefix in zip(per_request, prefix_lens)]
        )
        new_rows = torch.cat(
            [rows[prefix:] for rows, prefix in zip(per_request, prefix_lens)]
        )
        build_rows = sys.modules[
            "sglang.srt.hardware_backend.npu.prefill_dma"
        ].mapped_prefill_row_plans
        expected = build_rows(
            torch.cat((prefix_rows, new_rows)),
            prefix_rows.numel(),
            host_map,
            page_size,
            capacity,
        )
        plans = module.build_prefill_page_plans(
            pages, prefix_lens, seq_lens, host_map, page_size, capacity
        )
        for actual, reference in zip(plans, expected):
            torch.testing.assert_close(actual, reference)


def test_page_spans_reject_overlapping_writes(queued_chunk):
    case, _, _ = queued_chunk
    module = sys.modules[case.runtime.chunk_planner.__class__.__module__]
    with pytest.raises(ValueError, match="duplicate destination"):
        module.build_prefill_page_plans(
            torch.tensor([1, 2]),
            (1, 0),
            (3, 3),
            torch.tensor([0, 1, 1]),
            4,
            12,
        )


def test_discard_running_plan_keeps_worker_bounded(queued_chunk, monkeypatch):
    case, batch, submit = queued_chunk
    planner = case.runtime.chunk_planner
    module = sys.modules[planner.__class__.__module__]
    build = module.build_prefill_page_plans
    entered = ThreadEvent()
    release = ThreadEvent()

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return build(*args)

    monkeypatch.setattr(module, "build_prefill_page_plans", blocked)
    submit()
    try:
        assert entered.wait(5)
        previous = planner._inflight
        planner.discard()
        submit()
        assert planner._inflight is previous
        assert planner.take(batch, batch.req_pool_indices_cpu) is None
    finally:
        release.set()
    previous.result(timeout=5)
    submit()
    assert planner.take(batch, batch.req_pool_indices_cpu) is not None


def test_prefetch_does_not_wait_for_current_attention_completion(setup):
    case = setup()
    runtime = case.runtime
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[2],
        seq_lens_cpu=torch.tensor([4]),
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[2:4],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with pytest.raises(RuntimeError, match="stop test"):
        with runtime.forward(batch):
            runtime.prepare_layer(7)
            case.device.current.mark("scatter-1-0")
            runtime.on_kv_written(7, case.mapping[2:4])
            runtime.on_compute_start(7)
            restore_b = next(item for item in case.copies if item[:2] == ("restore", 1))
            assert "attention-1-0" not in restore_b[4]
            assert "save-1-0" not in restore_b[4]
            with pytest.raises(RuntimeError, match="Duplicate"):
                runtime.validate_write(7)
            raise RuntimeError("stop test")
    assert runtime.failed and not runtime.active
    with pytest.raises(RuntimeError, match="failed"):
        with runtime.forward(batch):
            pass


def test_missing_predecessor_hook_rejected_before_future_wait(setup):
    case = setup()
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        seq_lens_cpu=torch.tensor([2]),
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:2],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with pytest.raises(RuntimeError, match="not prefetched"):
        with case.runtime.forward(batch):
            case.runtime.prepare_layer(8)


def test_empty_write_generation_and_request_release(setup):
    case = setup()
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:0],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with case.runtime.forward(batch):
        with pytest.raises(RuntimeError, match="active"):
            case.runtime.release_request("request")
        for layer in range(7, 11):
            case.runtime.prepare_layer(layer)
            case.device.current.mark(f"scatter-1-{layer - 7}")
            case.runtime.on_compute_start(layer)
            case.runtime.finish_layer(layer)
    case.runtime.release_request("request")
    assert not case.runtime.retired and case.runtime.host_visible is None


def test_nonempty_generation_requires_real_scatter(setup):
    case = setup()
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        seq_lens_cpu=torch.tensor([2]),
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:2],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with pytest.raises(RuntimeError, match="KV-written"):
        with case.runtime.forward(batch):
            case.runtime.prepare_layer(7)
            case.runtime.on_compute_start(7)


def test_prefill_capture_rejected_before_copy_submission(setup):
    case = setup()
    case.device.capturing = True
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
    )
    with pytest.raises(RuntimeError, match="graph capture"):
        with case.runtime.forward(batch):
            pass
    assert not case.copies and not case.runtime.active


def test_decode_on_another_stream_waits_for_host_visibility(setup):
    case = setup()
    writer = case.device.current
    writer.mark("last-generation-published")
    case.runtime.host_visible = case.runtime._record(writer)
    decoder = case.device.Stream("cpu")
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: False)
    )
    with case.device.stream(decoder), case.runtime.forward(batch):
        assert "last-generation-published" in case.device.current.tail
        case.device.current.mark("decode-reader")
    ready = case.pool.decode_host_backing.page_binding.ready_event
    assert "decode-reader" in ready.dependencies
    assert not case.runtime.active


def test_layer_hook_cannot_change_compute_stream(setup):
    case = setup()
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        seq_lens_cpu=torch.tensor([2]),
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=case.mapping[:2],
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
    )
    with pytest.raises(RuntimeError, match="forward compute stream"):
        with case.runtime.forward(batch):
            with case.device.stream(case.device.Stream("cpu")):
                case.runtime.prepare_layer(7)
    assert case.runtime.failed and not case.runtime.active


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
