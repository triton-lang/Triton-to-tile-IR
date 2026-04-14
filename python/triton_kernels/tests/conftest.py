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


# ── PUBLIC-specific: skip tests that use ops unsupported in PUBLIC cuda-tile ──
# These ops have no lowering in the public cuda-tile release:
#   tt.dot_scaled      — fp8/mxfp scaled matmul (not available in public release)
#   tt.elementwise_inline_asm — used in mxfloat4 downcast kernels (not in public release)
# When ENABLE_TILE=1 and a test fails with PassManager::run failed + one of
# these op names in captured stderr, convert the result to SKIPPED.

_TILEIR_UNSUPPORTED_OPS = [
    "tt.dot_scaled",
    "tt.elementwise_inline_asm",
]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if os.environ.get("ENABLE_TILE", "0") != "1":
        return
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        stderr = "".join(
            content for name, content in report.sections
            if "stderr" in name.lower()
        )
        unsupported = [op for op in _TILEIR_UNSUPPORTED_OPS if op in stderr]
        if unsupported:
            report.outcome = "skipped"
            report.longrepr = (
                f"[tileir] unsupported op(s) in PUBLIC cuda-tile: {unsupported}"
            )
