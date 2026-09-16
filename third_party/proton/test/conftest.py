import os

import pytest

_RUNTIME_SELECTION_ENV = (
    "TRITON_LIBHIP_PATH",
    "TRITON_HSA_RUNTIME_PATH",
    "TRITON_HSA_RUNTIME_LIBRARY",
    "TRITON_ROCPROFILER_SDK_INCLUDE_PATH",
    "TRITON_ROCPROFILER_SDK_LIB_PATH",
    "TRITON_ROCPROFILER_SDK_LIBRARY",
    "TRITON_ROCTRACER_LIB_PATH",
    "TRITON_ROCTRACER_LIBRARY",
    "TRITON_ROCTX_LIB_PATH",
    "TRITON_ROCTX_LIBRARY",
)


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl

    # TheRock installs ROCm libraries outside the system loader paths. Proton
    # selects their absolute paths at import time, so keep those selections
    # while resetting mutable test knobs such as TRITON_PROTON_DISABLE.
    runtime_selection_env = {key: os.environ[key] for key in _RUNTIME_SELECTION_ENV if key in os.environ}
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        fresh = fresh_function()
        os.environ.update(runtime_selection_env)
        yield fresh
    finally:
        reset_function()

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
    # These complete tests require instrumentation. CUPTI-only API/profile
    # tests, including PC sampling and hw_trace, remain executable.
    unparametrized = {
        "test_jit", "test_select_ids", "test_trace", "test_multi_session",
        "test_autotune", "test_warp_spec", "test_timeline", "test_globaltime",
        "test_overhead", "test_gmem_buffer", "test_event_args", "test_threaded_kernel_call",
    }
    for item in items:
        params = item.callspec.params if hasattr(item, "callspec") else {}
        name = item.originalname
        path = item.path.resolve()
        unsupported = path == root / "test_override.py" and name == "test_override" and not params
        if path == root / "test_api.py" and name == "test_hook_manager" and not params:
            unsupported = True
        if path == root / "test_instrumentation.py":
            unsupported = name in unparametrized and not params
            if name == "test_mode_str" and set(params) == {"mode"}:
                unsupported = params["mode"] in (
                    "default", "default:metric_type=cycle", "default:metric_type=cycle:buffer_size=4096", "mma",
                )
            elif name == "test_mode_obj" and set(params) == {"mode"}:
                mode = item.module.proton.mode
                value = params["mode"]
                unsupported = (type(value) is mode.Default and value in (
                    mode.Default(), mode.Default(metric_type="cycle"),
                    mode.Default(metric_type="cycle", buffer_size=4096),
                )) or (type(value) is mode.MMA and value == mode.MMA())
            elif name == "test_record" and set(params) == {"method"}:
                unsupported = params["method"] in ("operator", "context_manager")
            elif name == "test_tree" and set(params) == {"hook"}:
                unsupported = params["hook"] in ("triton", None)
            elif name == "test_gluon_clc_profile" and set(params) == {"profile_data", "file_suffix"}:
                unsupported = (params["profile_data"], params["file_suffix"]) in (
                    ("tree", ".hatchet"), ("trace", ".chrome_trace"),
                )
            elif name == "test_tensor_descriptor" and set(params) == {"num_ctas"}:
                unsupported = type(params["num_ctas"]) is int and params["num_ctas"] in (1, 2)
        if unsupported:
            item.add_marker(pytest.mark.xfail(
                run=False, strict=True,
                reason="CTK 13.4 TileIR: Proton instrumentation is explicitly unsupported by profiler.start",
            ))
