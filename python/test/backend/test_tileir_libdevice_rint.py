"""Round-to-nearest-even through libdevice rint/nearbyint, including signed zero."""
import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_VALUES = [
    # Results that must be -0.0.
    -0.0, -0.25, -0.5, -1e-30,
    # Results that must be +0.0.
    0.0, 0.25, 0.5, 1e-30,
    # Ties round to the even neighbour.
    1.5, 2.5, 3.5, 4.5, -1.5, -2.5, -3.5, -4.5,
    8388606.5, 8388607.5, -8388606.5, -8388607.5,
    # Ordinary values on both sides of a tie.
    0.75, -0.75, 100.25, -100.75,
    # Already integral, beyond the fractional range of float32.
    1e10, -1e10, 7.0, -7.0,
    float("inf"), float("-inf"), float("nan"), 12345.5,
]


@triton.jit
def _rint(src, dst, NEARBY: tl.constexpr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    x = tl.load(src + offsets)
    if NEARBY:
        y = libdevice.nearbyint(x)
    else:
        y = libdevice.rint(x)
    tl.store(dst + offsets, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nearby", [False, True])
def test_libdevice_rint_signed_zero_and_ties(dtype, nearby):
    x = np.array(_VALUES, dtype=dtype)
    assert x.size == 32
    src = torch.from_numpy(x).cuda()
    dst = torch.empty_like(src)
    _rint[(1, )](src, dst, nearby, x.size)
    got = dst.cpu().numpy()
    expected = np.rint(x)
    np.testing.assert_array_equal(got, expected)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(np.signbit(got[finite]), np.signbit(expected[finite]))
