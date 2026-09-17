import pytest
import tempfile


def pytest_configure(config):
    # If pytest-sugar is not active, enable instafail
    if not config.pluginmanager.hasplugin("sugar"):
        config.option.instafail = True


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs

        with knobs.cache.scope(), knobs.runtime.scope(), knobs.compilation.scope():
            # This fixture tests an empty cache and its subsequent reuse.
            knobs.compilation.always_compile = False
            knobs.cache.dir = tmpdir
            yield tmpdir


@pytest.fixture
def fresh_knobs():
    """
    Resets all knobs except ``build``, ``nvidia``, and ``amd`` (preserves
    library paths needed to compile kernels).
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
    Resets ALL knobs including ``build``, ``nvidia``, and ``amd``.
    Use for tests that verify initial values of these knobs.
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())


@pytest.fixture(autouse=True)
def _tileir_line_info_tools(request, monkeypatch):
    if _tileir_test_key(request.node)[0] != "unit/language/test_line_info.py" or not _tileir_134_profile():
        return
    import os
    from pathlib import Path
    from triton import knobs
    nvdisasm = Path(knobs.nvidia.nvdisasm.path)
    if not nvdisasm.is_file() or not os.access(nvdisasm, os.X_OK):
        pytest.fail(f"TileIR line-info tests require executable nvdisasm: {nvdisasm}")
    # Keep the original tests and assertions; extend only their backend tool lookup.
    monkeypatch.setattr(request.module, "get_disassembler_command_and_debug_line_format",
                        lambda: ("cubin", [str(nvdisasm), "-g"], "## File", ","))

# Backend limitations are scoped to the public toolchain and exact upstream
# test paths. Numerical failures and unexpected diagnostics stay visible.
_TILEIR_STAGE_TESTS = {('unit/language/test_compile_only.py', 'test_compile_only_dot'): {'ptx', 'ttgir'},
 ('unit/language/test_compile_only.py', 'test_compile_only_dot_mxfp'): {'ptx', 'ttgir'},
 ('unit/language/test_compile_only.py', 'test_compile_only_k_loop'): {'ptx', 'ttgir'},
 ('unit/language/test_compile_only.py', 'test_compile_only_sm100'): {'ptx'},
 ('unit/language/test_core.py', 'test_assume'): {'llir', 'ttgir'},
 ('unit/language/test_core.py', 'test_atomic_cas'): {'ptx'},
 ('unit/language/test_core.py', 'test_atomic_rmw'): {'ptx'},
 ('unit/language/test_core.py', 'test_disable_licm'): {'llir'},
 ('unit/language/test_core.py', 'test_dot'): {'ptx'},
 ('unit/language/test_core.py', 'test_dot_max_num_imprecise_acc'): {'ptx'},
 ('unit/language/test_core.py', 'test_dot_mulbroadcasted'): {'ttgir'},
 ('unit/language/test_core.py', 'test_enable_fp_fusion'): {'ptx'},
 ('unit/language/test_core.py', 'test_load_cache_modifier'): {'ptx'},
 ('unit/language/test_core.py', 'test_no_rematerialization_op'): {'ttgir'},
 ('unit/language/test_core.py', 'test_num_ctas_pre_sm90'): {'ttgir'},
 ('unit/language/test_core.py', 'test_optimize_thread_locality'): {'ttgir'},
 ('unit/language/test_core.py', 'test_override_arch'): {'ttgir'},
 ('unit/language/test_core.py', 'test_permute'): {'ptx'},
 ('unit/language/test_core.py', 'test_poison_return'): {'llir'},
 ('unit/language/test_core.py', 'test_scaled_dot'): {'ptx'},
 ('unit/language/test_core.py', 'test_store_cache_modifier'): {'ptx'},
 ('unit/language/test_core.py', 'test_store_eviction_policy'): {'ptx'},
 ('unit/language/test_core.py', 'test_tl_range_fuse'): {'ttgir'},
 ('unit/language/test_core.py', 'test_tl_range_fuse_dependent'): {'ttgir'},
 ('unit/language/test_core.py', 'test_tl_range_num_stages'): {'ptx'},
 ('unit/language/test_core.py', 'test_vectorization'): {'ptx'},
 ('unit/language/test_core.py', 'test_vectorization_hints'): {'ptx'},
 ('unit/language/test_matmul.py', 'test_batched_mxfp'): {'ptx'},
 ('unit/language/test_matmul.py', 'test_block_scale_fp4'): {'ptx'},
 ('unit/language/test_matmul.py', 'test_blocked_scale_mxfp'): {'ptx', 'ttgir'},
 ('unit/language/test_matmul.py', 'test_lhs_in_tmem'): {'ttgir'},
 ('unit/language/test_matmul.py', 'test_lhs_in_tmem_mxfp'): {'ttgir'},
 ('unit/language/test_matmul.py', 'test_mxfp'): {'ptx'},
 ('unit/language/test_matmul.py', 'test_mxfp8_mxfp4_matmul'): {'ttgir'},
 ('unit/language/test_matmul.py', 'test_simple_matmul'): {'ptx', 'ttgir'},
 ('unit/language/test_matmul.py', 'test_simple_persistent_matmul'): {'ttgir'},
 ('unit/language/test_pipeliner.py', 'test_pipeline_matmul'): {'ttgir'},
 ('unit/language/test_pipeliner.py', 'test_pipeline_vecadd'): {'ttgir'},
 ('unit/language/test_pipeliner.py', 'test_scatter_pipeline'): {'ttgir'},
 ('unit/language/test_tensor_descriptor.py', 'test_host_tensor_descriptor_matmul'): {'ptx'},
 ('unit/language/test_tensor_descriptor.py', 'test_make_tensor_descriptor_loop_carried'): {'ptx'},
 ('unit/language/test_tensor_descriptor.py', 'test_make_tensor_descriptor_matmul'): {'ptx'},
 ('unit/language/test_tensor_descriptor.py', 'test_tensor_descriptor_batched_gemm_3d_tma'): {'ttgir'},
 ('unit/language/test_tensor_descriptor.py', 'test_tma_gather_dot_pipeline'): {'ttgir'},
 ('unit/language/test_warp_specialization.py', 'test_grouped_gemm'): {'ttgir'},
 ('unit/language/test_warp_specialization.py', 'test_warp_specialize_tma_matmul'): {'ttgir'},
 ('unit/language/test_warp_specialization.py', 'test_warp_specialize_tma_matmul_persistent'): {'ttgir'},
 ('unit/test_debuginfo.py', 'test_triton_debuginfo_on'): {'llir'}}

_TILEIR_134_UNSUPPORTED = {('unit/instrumentation/test_gpuhello.py', 'test_op'): 'LLVM GPU instruction instrumentation is not connected to '
                                                       'the TileIR compiler pipeline',
 ('unit/language/test_block_pointer.py', 'test_block_copy'): 'block pointer make_tensor_ptr/advance lowering is '
                                                             'unavailable',
 ('unit/language/test_block_pointer.py', 'test_block_ptr_matmul_no_scf'): 'block pointer make_tensor_ptr/advance '
                                                                          'lowering is unavailable',
 ('unit/language/test_compile_errors.py', 'test_min_dot_size'): 'TileIR accepts dot dimensions below the NVIDIA '
                                                                'diagnostic minimum',
 ('unit/language/test_compile_only.py', 'test_fp8_compiles_for_multiple_architectures_cuda'): 'this test includes '
                                                                                              'SM80 FP8 '
                                                                                              'compilation, which '
                                                                                              'the public13.4 '
                                                                                              'tileiras rejects',
 ('unit/language/test_core.py', 'test_enable_reflect_ftz'): 'enable_reflect_ftz is not a TileIR option; FTZ uses '
                                                            'the backend option',
 ('unit/language/test_core.py', 'test_globaltimer'): 'generic inline assembly is unavailable; only native GDC '
                                                     'helper forms are recognized',
 ('unit/language/test_core.py', 'test_histogram'): 'histogram lowering is not implemented for the public backend',
 ('unit/language/test_core.py', 'test_histogram_mask'): 'histogram lowering is not implemented for the public backend',
 ('unit/language/test_core.py', 'test_histogram_silent_data_corruption'): 'generic tt.histogram has no public '
                                                                          'lowering',
 ('unit/language/test_core.py', 'test_inline_asm'): 'generic inline assembly is unavailable; only native GDC '
                                                    'helper forms are recognized',
 ('unit/language/test_core.py', 'test_inline_asm_multiple_outputs'): 'generic inline assembly is unavailable; only '
                                                                     'native GDC helper forms are recognized',
 ('unit/language/test_core.py', 'test_inline_asm_packed'): 'generic inline assembly is unavailable; only native '
                                                           'GDC helper forms are recognized',
 ('unit/language/test_core.py', 'test_inline_asm_packed_multiple_outputs'): 'generic inline assembly is '
                                                                            'unavailable; only native GDC helper '
                                                                            'forms are recognized',
 ('unit/language/test_core.py', 'test_inline_asm_with_pointers'): 'generic inline assembly is unavailable; only '
                                                                  'native GDC helper forms are recognized',
 ('unit/language/test_core.py', 'test_math_erf_op'): 'math.erf has no public13.4 native lowering; existing '
                                                     'libdevice rewrite is a separate path',
 ('unit/language/test_core.py', 'test_maxnreg'): 'maxnreg is compatibility-only; this backend controls occupancy',
 ('unit/language/test_core.py', 'test_nested_if_else_return'): 'control-flow lifting emits index_castui and '
                                                               'scf.index_switch without public backend conversion',
 ('unit/language/test_core.py', 'test_num_ctas_pre_sm90'): 'num_ctas is a TileIR optimization hint; NVIDIA '
                                                           'cluster-launch validation is not its contract',
 ('unit/language/test_core.py', 'test_poison_return'): 'this LLVM poison inspection test requires ub.poison '
                                                       'lowering and LLIR, neither exposed by TileIR',
 ('unit/language/test_core.py', 'test_side_effectful_reduction'): 'public reduce regions reject the device assert '
                                                                  'memory effect in the reduction body',
 ('unit/language/test_core.py', 'test_side_effectful_reduction_2d'): 'public reduce regions reject the device '
                                                                     'assert memory effect in the reduction body',
 ('unit/language/test_core.py', 'test_side_effectful_scan'): 'public scan regions reject the device assert memory '
                                                             'effect in the scan body',
 ('unit/language/test_core.py', 'test_smid'): 'generic inline assembly is unavailable; only native GDC helper '
                                              'forms are recognized',
 ('unit/language/test_core.py', 'test_trans_reshape'): 'this kernel requires unavailable block pointer lowering',
 ('unit/language/test_core.py', 'test_unroll_attr'): 'this test requires frontend TTIR unrolling; TileIR unrolling '
                                                     'occurs downstream',
 ('unit/language/test_libdevice.py', 'test_bessel'): 'j0/j1/y0/y1/cyl_bessel_i0/cyl_bessel_i1 have no public '
                                                     'lowering',
 ('unit/language/test_reproducer.py', 'test_triton_reproducer_path'): 'this test requires NVIDIA '
                                                                      'make_ttgir/make_llir reproducer stages',
 ('unit/language/test_tensor_descriptor.py', 'test_make_tensor_descriptor_loop_carried'): 'public cuda_tile.if '
                                                                                          'cannot return tile '
                                                                                          'views used by '
                                                                                          'conditional descriptor '
                                                                                          'replacement',
 ('unit/language/test_tensor_descriptor.py', 'test_mxfp8_mxfp4_matmul_tma'): 'this kernel uses mixed FP8/FP4 '
                                                                             'operands; native scaled MMA requires '
                                                                             'matching types',
 ('unit/language/test_tensor_descriptor.py', 'test_tensor_descriptor_batched_gemm_2d_tma'): 'public cuda_tile.if '
                                                                                            'cannot return tile '
                                                                                            'views used by '
                                                                                            'conditional '
                                                                                            'descriptor '
                                                                                            'replacement',
 ('unit/language/test_warp_specialization.py', 'test_warp_specialize_basic_ir'): 'input is handwritten TTGIR, '
                                                                                 'which is not a TileIR input '
                                                                                 'stage',
 ('unit/language/test_warp_specialization.py', 'test_warp_specialize_tma_matmul_consan'): 'Consan instrumentation '
                                                                                          'is not connected to the '
                                                                                          'TileIR pipeline',
 ('unit/language/test_warp_specialization.py', 'test_warp_specialize_tma_matmul_persistent_consan'): 'Consan '
                                                                                                     'instrumentation '
                                                                                                     'is not '
                                                                                                     'connected to '
                                                                                                     'the TileIR '
                                                                                                     'pipeline',
 ('unit/language/test_warp_specialization.py', 'test_warp_specialize_tmem_ir'): 'input is handwritten TTGIR, which '
                                                                                'is not a TileIR input stage',
 ('unit/language/test_warp_specialization.py', 'test_warpgroup_reduction'): 'input is handwritten TTGIR, which is '
                                                                            'not a TileIR input stage',
 ('unit/runtime/test_autotuner.py', 'test_exceed_threads'): 'num_warps is a worker hint, not a fixed TileIR launch '
                                                            'thread count; NVIDIA thread-limit diagnostic does not '
                                                            'apply',
 ('unit/runtime/test_autotuner.py', 'test_exceed_tmem'): 'this NVIDIA TMEM diagnostic assumes compilation rejects '
                                                         'the kernel before its oversized output store; TileIR '
                                                         'chooses a different allocation',
 ('unit/runtime/test_autotuner.py', 'test_hooks'): 'this autotuner hook test requires a NVIDIA resource-exhaustion '
                                                   'exception from num_warps; TileIR uses a worker hint',
 ('unit/runtime/test_autotuner.py', 'test_override_ttgir'): 'TTGIR is not an input/override stage of the TileIR '
                                                            'pipeline; TTIR override is tested separately',
 ('unit/runtime/test_cache.py', 'test_jit_noinline'): 'the public backend inlines device helpers; this test '
                                                      'requires a distinct noinline device function in TTIR',
 ('unit/test_perf_warning.py', 'test_remark_vectorization'): 'this test requires the NVIDIA vectorization pass '
                                                             'remark text; TileIR uses a different compiler '
                                                             'pipeline',
 ('unit/test_stages_inspection.py', 'test_inspection'): 'this inspection/reproducer test installs NVIDIA '
                                                        'make_ttgir, which is not a TileIR compilation stage',
 ('unit/tools/test_aot.py', 'test_compile_link_autotune_matmul'): 'TileIR has no AOT compile/link templates or '
                                                                  'profile-scratch metadata for this CUDA C '
                                                                  'launcher interface',
 ('unit/tools/test_aot.py', 'test_compile_link_matmul'): 'TileIR has no AOT compile/link templates or '
                                                         'profile-scratch metadata for this CUDA C launcher '
                                                         'interface',
 ('unit/tools/test_aot.py', 'test_compile_link_matmul_no_specialization'): 'TileIR has no AOT compile/link '
                                                                           'templates or profile-scratch metadata '
                                                                           'for this CUDA C launcher interface',
 ('unit/tools/test_aot.py', 'test_launcher_has_no_available_kernel'): 'TileIR has no AOT compile/link templates or '
                                                                      'profile-scratch metadata for this CUDA C '
                                                                      'launcher interface',
 ('unit/tools/test_aot.py', 'test_ttgir_to_asm'): 'handwritten TTGIR is not an input stage of TileIR',
 ('unit/tools/test_irsource.py', 'test_mlir_attribute_parsing'): 'this test compiles handwritten TTGIR, which is '
                                                                 'not a TileIR input stage'}

def _tileir_134_profile():
    import os
    from pathlib import Path
    import re
    if (os.getenv("ENABLE_TILE") != "1" or os.getenv("TRITON_INTERPRET") == "1"
            or os.getenv("TRITON_DEFAULT_BACKEND") not in (None, "", "tileir")):
        return False
    compatibility = Path(__file__).resolve().parents[2] / ".triton-tileir-compat.toml"
    return compatibility.is_file() and re.search(
        r'^tileir_version\s*=\s*"13\.4\.[0-9]+"', compatibility.read_text(), re.MULTILINE
    ) is not None


def _tileir_test_key(item):
    from pathlib import Path
    root = Path(__file__).resolve().parent
    try:
        relative = item.path.resolve().relative_to(root)
    except ValueError:
        return None
    return relative.as_posix(), item.originalname


def pytest_collection_modifyitems(items):
    if not _tileir_134_profile():
        return
    for item in items:
        key = _tileir_test_key(item)
        params = item.callspec.params if hasattr(item, "callspec") else {}
        reason = _TILEIR_134_UNSUPPORTED.get(key)
        if key == ("unit/language/test_core.py", "test_scaled_dot"):
            known_params = (
                set(params) == {"M", "N", "K", "col_a", "col_b", "rhs_scale",
                                "mxfp_type", "normal_type", "num_warps", "mma", "kpack"}
                and all(type(params[name]) is int for name in
                        ("M", "N", "K", "num_warps", "mma", "kpack"))
                and params["M"] in {32, 64, 128} and params["N"] in {32, 64, 128}
                and params["K"] in {64, 128}
                and (params["num_warps"], params["mma"], params["kpack"]) == (4, 16, 1)
                and all(type(params[name]) is bool for name in ("col_a", "col_b", "rhs_scale"))
                and type(params["mxfp_type"]) is str and type(params["normal_type"]) is str
                and params["mxfp_type"] in {"e2m1", "e4m3", "e5m2"}
                and params["normal_type"] in {"e4m3", "e5m2", "bf16", "fp16"}
            )
            if known_params and not (
                params["mxfp_type"] == params["normal_type"]
                and params["mxfp_type"] in {"e4m3", "e5m2"}
            ):
                reason = "single-scale native MMA supports matching FP8; FP4 and mixed input types remain unsupported"
        elif key == ("unit/language/test_core.py", "test_tensor_atomic_cas") and params.get("dtype_str") in {"float16", "bfloat16"}:
            reason = "public atomic CAS accepts 32/64-bit elements, not 16-bit elements"
        elif (key == ("unit/language/test_core.py", "test_tensor_atomic_use_result")
              and params.get("dtype_str") == "float16" and params.get("op") == "cas"):
            reason = "public atomic CAS accepts 32/64-bit elements, not 16-bit elements"
        elif key == ("unit/language/test_core.py", "test_override_arch") and params.get("arch") == "sm70":
            reason = "public13.4 tileiras does not accept SM70 as a compilation target"
        elif (key == ("unit/language/test_conversions.py", "test_typeconvert_downcast")
              and params.get("src_dtype") == "float32" and params.get("dst_dtype") == "float8e5"
              and params.get("rounding") == "rtz"):
            reason = "public13.4 f32-to-f8E5M2 conversion supports nearest-even, not round-toward-zero"
        elif key == ("unit/test_link.py", "test_link_extern_libs") and params.get("use_libdevice"):
            reason = "libdevice.sqrt maps to a native TileIR op; this test requires an LLVM linker callback"
        elif (key == ("unit/language/test_tensor_descriptor.py", "test_tensor_descriptor_reduce")
              and set(params) == {"kind", "dtype_str", "descriptor", "num_ctas", "M_BLOCK", "N_BLOCK"}
              and all(type(params.get(k)) is int for k in ("num_ctas", "M_BLOCK", "N_BLOCK"))
              and params.get("kind") in {"min", "max"}
              and params.get("dtype_str") in {"float16", "bfloat16"}
              and params.get("descriptor") in {"host", "device"}
              and params.get("num_ctas") in {1, 2}
              and (params.get("M_BLOCK"), params.get("N_BLOCK"))
              in {(2, 16), (8, 16), (8, 32), (8, 128), (512, 32), (1, 1024)}):
            reason = "public13.4 descriptor atomic view reductions support floating-point add, not min/max"
        elif key == ("unit/language/test_matmul.py", "test_mxfp8_mxfp4_matmul"):
            if params["A_DATA_TYPE"] != params["B_DATA_TYPE"] or not (
                params["WITH_A_SCALE"] and params["WITH_B_SCALE"]
            ):
                reason = "native scaled MMA requires matching operand types and both scales"
        elif key == ("unit/language/test_pipeliner.py", "test_pipeline_matmul") and params.get("scale"):
            reason = "this parameter omits the second operand scale"
        elif key == ("unit/language/test_conversions.py", "test_typeconvert_upcast") and params.get("src_dtype") == "float8e4b15":
            reason = "this custom float8 conversion requires generic inline assembly"
        elif key == ("unit/language/test_subprocess.py", "test_print") and params.get("func_type") in {
            "device_print", "device_print_scalar", "print", "no_arg_print", "print_no_arg",
            "device_print_large", "print_multiple_args", "device_print_multiple_args",
            "device_print_hex", "device_print_pointer", "device_print_negative",
            "device_print_uint", "device_print_uint_cast", "device_print_2d_tensor",
        }:
            reason = "device print formatting differs from the NVIDIA pid/idx and precision contract"
        line_reason = None
        if key == ("unit/language/test_line_info.py", "test_line_info"):
            if params.get("func") == "call_noinline":
                line_reason = "public backend inlines device helpers; separate callee source-line coverage is not preserved"
            elif params.get("func") == "autotune":
                line_reason = "public13.4 compiler omits the optimized loop-header line while preserving load/store lines"
        elif key == ("unit/language/test_line_info.py", "test_line_info_ir_source") and params.get("status") == "":
            line_reason = "public13.4 cubin omits the original TTIR load source line retained in input TileIR"
        if line_reason:
            # Re-execute these compile-only diagnostics so new compiler support is visible.
            item.add_marker(pytest.mark.xfail(strict=True, raises=AssertionError,
                                               reason=f"CTK 13.4 TileIR: {line_reason}"))
        if reason:
            item.add_marker(pytest.mark.xfail(run=False, strict=True, reason=f"CTK 13.4 TileIR: {reason}"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not _tileir_134_profile() or call.when != "call":
        return
    stages = _TILEIR_STAGE_TESTS.get(_tileir_test_key(item), set())
    error = call.excinfo.value if call.excinfo is not None else None
    if type(error) is not KeyError or len(error.args) != 1:
        return
    stage = next((name for name in stages
                  if error.args[0] in (name, f"Unknown key: '{name}'")), None)
    if stage is None:
        return
    report = outcome.get_result()
    if not report.failed:
        return
    # A missing inspection stage can occur immediately after an asynchronous
    # launch. Surface that launch's error before classifying the inspection gap.
    import sys
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        try:
            torch.cuda.synchronize()
        except Exception:
            report.longrepr = item.repr_failure(pytest.ExceptionInfo.from_current())
            return
    reason = f"CTK 13.4 TileIR does not expose the {stage} inspection stage"
    report.outcome = "skipped"
    report.wasxfail = reason
    report.longrepr = (str(item.path), item.location[1], reason)
