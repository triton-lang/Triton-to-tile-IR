import pytest
import tempfile
import os


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_knobs():
    """
    Default fresh knobs fixture that preserves library path
    information from the environment as these are typically
    needed to successfully compile kernels.
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_including_libraries():
    """
    A variant of `fresh_knobs` that resets ALL knobs including
    library paths. Use this only for tests that need complete
    environment isolation.
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs

        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


def pytest_configure(config):
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None and worker_id.startswith("gw"):
        import torch
        gpu_id = int(worker_id[2:])  # map gw0 → 0, gw1 → 1, ...
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id % torch.cuda.device_count())

def _tileir_134_profile():
    import os
    import re
    from pathlib import Path
    if (os.getenv("ENABLE_TILE") != "1" or os.getenv("TRITON_INTERPRET") == "1"
            or os.getenv("TRITON_DEFAULT_BACKEND") not in (None, "", "tileir")):
        return False
    compatibility = Path(__file__).resolve().parents[3] / ".triton-tileir-compat.toml"
    return compatibility.is_file() and re.search(
        r'^tileir_version\s*=\s*"13\.4\.[0-9]+"', compatibility.read_text(), re.MULTILINE
    ) is not None


def pytest_collection_modifyitems(items):
    from pathlib import Path
    if not _tileir_134_profile():
        return
    root = Path(__file__).resolve().parent
    path = root / "test_matmul.py"
    # The FP8 cases supply a unit LHS scale; BF16 cases omit it. Both kernel
    # variants call dot_scaled for these pairs without value swizzling.
    unsupported_pairs = {
        ("bfloat16", "mxfloat4_e2m1"),
        ("bfloat16", "mxfloat8_e4m3fn"),
        ("float8_e5m2", "mxfloat4_e2m1"),
        ("float8_e5m2", "mxfloat8_e4m3fn"),
        ("mxfloat8_e4m3fn", "mxfloat4_e2m1"),
    }
    for item in items:
        params = item.callspec.params if hasattr(item, "callspec") else {}
        reason = None
        if (item.path.resolve() == root / "test_matmul_details/test_opt_flags_nvidia.py"
                and item.originalname in ("test_matmul_blackwell_scale_small_n",
                                          "test_matmul_blackwell_shuffled_mxfp4_weight")
                and not params):
            reason = "CTK 13.4 TileIR: native scaled MMA requires matching FP4/FP8 types and both scales"
        elif (item.path.resolve() == root / "test_tensor_details/test_layout_hopper.py"
              and item.originalname == "test_upcast_mxfp4_to_bf16"
              and set(params) == {"mx_axis", "num_warps"}
              and type(params["mx_axis"]) is int and params["mx_axis"] in (0, 1)
              and type(params["num_warps"]) is int and params["num_warps"] in (4, 8)):
            reason = "CTK 13.4 TileIR: Hopper MXFP4 unpacking requires unsupported packed BF16 inline assembly"
        if reason:
            item.add_marker(pytest.mark.xfail(run=False, strict=True, reason=reason))
            continue
        if item.path.resolve() != path or item.originalname != "test_op":
            continue
        pair = (params.get("act_dtype_str"), params.get("weight_dtype_str"))
        if pair not in unsupported_pairs:
            continue
        if (params.get("mode") not in ("plain", "batched", "ragged")
                or type(params.get("is_persistent")) is not bool
                or not all(type(params.get(dim)) is int and params[dim] > 0 for dim in ("m", "n", "k"))):
            continue
        if type(params.get("b_hbm_swizzling")) is not bool:
            continue
        if params["b_hbm_swizzling"]:
            # The existing layout selector picks Blackwell value layout on
            # SM100. Hopper's BF16 decode + ordinary dot must remain runnable.
            target = item.module.triton.runtime.driver.active.get_current_target()
            if target.backend != "tileir" or target.arch != 100:
                continue
        item.add_marker(pytest.mark.xfail(
            run=False, strict=True,
            reason="CTK 13.4 TileIR: native scaled MMA requires matching FP4/FP8 types and both scales",
        ))
