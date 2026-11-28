"""Drain service users before closing the NPU Host allocation owners."""


def _agree_on_shutdown(coordinators, error):
    groups = {}
    for coordinator in coordinators:
        backing = coordinator._backing
        pool = backing.pool if backing is not None else None
        shared = getattr(pool, "shared_group", None)
        if shared is not None:
            group = shared.group
            groups[id(getattr(group, "cpu_group", group))] = group
    for group in groups.values():
        errors = group.all_gather_object(str(error) if error is not None else None)
        if any(item is not None for item in errors):
            raise RuntimeError(
                f"NPU HiSparse shutdown could not drain all TP ranks: {errors}"
            ) from error
    if error is not None:
        raise error


def release_npu_hisparse_resources(scheduler):
    if getattr(scheduler, "_npu_hisparse_resources_released", False):
        return
    coordinators = [scheduler.hisparse_coordinator]
    draft = scheduler.draft_worker
    if draft is not None:
        for runner in draft._draft_model_runners():
            coordinator = runner.hisparse_coordinator
            if coordinator is not None and coordinator not in coordinators:
                coordinators.append(coordinator)
    error = None
    try:
        for coordinator in coordinators:
            coordinator.prepare_shutdown()
    except Exception as caught:
        error = caught
    _agree_on_shutdown(coordinators, error)
    error = None
    try:
        scheduler.tree_cache.release_host_resources()
    except Exception as caught:
        error = caught
    _agree_on_shutdown(coordinators, error)
    for coordinator in coordinators:
        coordinator.destroy()
    scheduler._npu_hisparse_resources_released = True
