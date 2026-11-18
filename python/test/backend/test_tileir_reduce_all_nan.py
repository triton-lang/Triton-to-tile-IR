"""NaN-ignoring min/max reductions must still return NaN when every element is NaN."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _reduce(src, dst, ROWS: tl.constexpr, COLS: tl.constexpr, AXIS: tl.constexpr, OUT: tl.constexpr,
            IS_MAX: tl.constexpr):
    offsets = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
    x = tl.load(src + offsets)
    if IS_MAX:
        y = tl.max(x, AXIS)
    else:
        y = tl.min(x, AXIS)
    tl.store(dst + tl.arange(0, OUT), y.to(src.dtype.element_ty))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("axis, rows, cols", [(0, 8, 16), (0, 16, 16), (0, 32, 4), (1, 8, 16), (1, 16, 16), (1, 2, 16)])
def test_nan_ignoring_reduction_of_all_nan_lane(dtype, op, axis, rows, cols):
    torch.manual_seed(0)
    x = torch.randint(-5, 6, (rows, cols), device="cuda").to(dtype)
    # One row and one column hold nothing but NaN; every other lane mixes NaN
    # with ordinary values.
    x[0, :] = torch.nan
    x[:, 0] = torch.nan
    out = cols if axis == 0 else rows
    dst = torch.full((out, ), 123.0, device="cuda", dtype=dtype)
    _reduce[(1, )](x, dst, rows, cols, axis, out, op == "max")

    isnan = torch.isnan(x)
    if op == "max":
        expected = x.masked_fill(isnan, -torch.inf).amax(axis)
    else:
        expected = x.masked_fill(isnan, torch.inf).amin(axis)
    expected = torch.where(isnan.all(axis), torch.full_like(expected, torch.nan), expected)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0, equal_nan=True)
