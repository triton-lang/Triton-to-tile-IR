"""PUBLIC backend integration regressions for the shared autotuner."""
import torch
import triton
import triton.language as tl
import pytest


def test_config_identity_does_not_access_device(monkeypatch):
    from triton.runtime.driver import driver

    class NoDeviceAccess:
        def get_current_target(self):
            raise AssertionError("Config hashing/equality must not query the device")

    monkeypatch.setattr(driver, "_active", NoDeviceAccess())
    first = triton.Config({"BLOCK": 32}, opt_level=2)
    equal = triton.Config({"BLOCK": 32}, opt_level=2)
    different = triton.Config({"BLOCK": 32}, opt_level=3)
    lookup = {first: "kept"}
    assert lookup[equal] == "kept"
    assert different not in lookup
    assert first != object()


def test_tileir_rejects_unimplemented_options():
    from triton.backends.compiler import GPUTarget
    from triton.backends.tileir.compiler import TileIRBackend

    backend = TileIRBackend(GPUTarget("tileir", 100, 32))
    assert backend.parse_options({"clc": False, "instrumentation_mode": ""}).clc is False
    for options, message in [({"clc": True}, "CLC"), ({"instrumentation_mode": "fpsan"}, "instrumentation")]:
        with pytest.raises(NotImplementedError, match=message):
            backend.parse_options(options)


def test_autotune_print_after_benchmark(device, fresh_knobs, capsys):
    fresh_knobs.autotuning.print = True
    launches = []

    def benchmark(call, quantiles):
        call()
        launches.append(1)
        return [1.0, 1.0, 1.0]

    @triton.autotune(configs=[triton.Config({"BLOCK": 32}), triton.Config({"BLOCK": 64})],
                     key=["N"], do_bench=benchmark)
    @triton.jit
    def copy(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        tl.store(dst + offsets, tl.load(src + offsets, offsets < N, 0), offsets < N)

    src = torch.arange(32, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    copy[(1,)](src, dst, src.numel())
    torch.testing.assert_close(dst, src)
    assert len(launches) == 2
    assert "best config selected" in capsys.readouterr().out


def test_attention_autotune_causal_switch(monkeypatch):
    """A non-causal tuning result must not bypass causal config pruning."""
    import importlib.util
    from pathlib import Path
    import sys

    path = Path(__file__).resolve().parents[2] / "tutorials/06-fused-attention.py"
    spec = importlib.util.spec_from_file_location("tileir_attention_autotune_regression", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._attn_fwd.configs = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=2, num_warps=4,
                      pre_hook=module._host_descriptor_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4,
                      pre_hook=module._host_descriptor_pre_hook),
    ]
    launches = []

    def benchmark(call, quantiles):
        call()
        launches.append(1)
        return [1.0, 1.0, 1.0]

    module._attn_fwd._do_bench = benchmark
    torch.manual_seed(20)
    q, k, v = [torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.float16) * 0.5 for _ in range(3)]
    mask = torch.ones((128, 128), device="cuda", dtype=torch.bool).tril()
    for causal in (False, True, False):
        expected = torch.matmul(q, k.transpose(2, 3)) * 0.5
        if causal:
            expected = expected.masked_fill(~mask, float("-inf"))
        expected = torch.matmul(torch.softmax(expected.float(), dim=-1).half(), v)
        actual = module.attention(q, k, v, causal, 0.5, False)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=0)
        if causal:
            launches_after_causal = len(launches)
    assert len(launches) == launches_after_causal
    assert len(launches) >= 2
