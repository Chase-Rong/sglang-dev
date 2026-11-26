"""External planner and V4 sparse attention bindings."""

SHARED_QUERY_HEADS = (16, 32)


def native_ops():
    from sgl_kernel_npu.hisparse import load_fsa

    return load_fsa()


def shared_workspace_bytes():
    """Return the actual device workspace bound for the fused execution."""
    return int(native_ops().shared_workspace_bytes())
