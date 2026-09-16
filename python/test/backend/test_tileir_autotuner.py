"""PUBLIC backend integration regressions for the shared autotuner."""
import torch
import triton
import triton.language as tl


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
