"""PUBLIC backend regressions for the max_occupancy tuning key."""
import pytest
import torch
import triton
import triton.language as tl


def test_max_occupancy_options():
    from triton.backends.compiler import GPUTarget
    from triton.backends.tileir.compiler import TileIRBackend

    backend = TileIRBackend(GPUTarget("tileir", 100, 32))
    for value in (None, 1, 2, 4, 32, 33, 1024):
        canonical = backend.parse_options({"max_occupancy": value})
        earlier = backend.parse_options({"occupancy": value})
        assert canonical.max_occupancy == earlier.max_occupancy == value
        assert canonical.occupancy is None and earlier.occupancy is None
        assert canonical.occupancy_hint == earlier.occupancy_hint == min(value or 1, 32)
        assert canonical.hash() == earlier.hash()
        assert backend.parse_options(vars(earlier)).hash() == canonical.hash()
    assert backend.parse_options({"occupancy": 2, "max_occupancy": 2}).max_occupancy == 2
    for key in ("max_occupancy", "occupancy"):
        for value in (0, -1, 1.5, True, "2"):
            with pytest.raises(ValueError, match=f"{key} must be a positive integer or None"):
                backend.parse_options({key: value})
    with pytest.raises(ValueError, match="occupancy and max_occupancy must agree"):
        backend.parse_options({"occupancy": 2, "max_occupancy": 4})


@pytest.mark.parametrize("max_occupancy", [None, 1, 2, 4, 1024])
def test_max_occupancy_hint(max_occupancy, device):

    @triton.jit
    def kernel(output):
        offsets = tl.arange(0, 32)
        tl.store(output + offsets, 7)

    output = torch.empty(32, device=device, dtype=torch.int32)
    compiled = kernel.warmup(output, grid=(1, ), max_occupancy=max_occupancy)
    assert compiled.metadata.max_occupancy == max_occupancy
    assert f"occupancy = {min(max_occupancy or 1, 32)}" in compiled.asm["tileir"]
    earlier = kernel.warmup(output, grid=(1, ), occupancy=max_occupancy)
    assert earlier.hash == compiled.hash
    compiled[(1, 1, 1)](output)
    torch.testing.assert_close(output, torch.full_like(output, 7))


def test_max_occupancy_autotune_key(device):

    def benchmark(call, quantiles):
        call()
        return [1.0] * len(quantiles)

    @triton.autotune(configs=[triton.Config({"max_occupancy": value}) for value in (1, 2)], key=[],
                     do_bench=benchmark)
    @triton.jit
    def kernel(output):
        offsets = tl.arange(0, 32)
        tl.store(output + offsets, 9)

    output = torch.empty(32, device=device, dtype=torch.int32)
    kernel[(1, )](output)
    assert {config.kwargs["max_occupancy"] for config in kernel.configs_timings} == {1, 2}
    torch.testing.assert_close(output, torch.full_like(output, 9))
