"""PTX div.full.f32 inline assembly on the TileIR backend.

The form is lowered to the native full-range f32 division, which is also what
an ordinary f32 ``x / y`` uses, so both spellings must agree bit for bit.
"""
import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver

BLOCK = 1024


@triton.jit
def _divide(X, Y, Z, INLINE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets).to(tl.float32, bitcast=True)
    y = tl.load(Y + offsets).to(tl.float32, bitcast=True)
    if INLINE:
        result = tl.inline_asm_elementwise("div.full.f32 $0, $1, $2;", "=f,f,f", [x, y], dtype=tl.float32,
                                           is_pure=True, pack=1)
    else:
        result = x / y
    tl.store(Z + offsets, result.to(tl.int32, bitcast=True))


@triton.jit
def _divide_by_constant(X, Z, DENOMINATOR: tl.constexpr, INLINE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets).to(tl.float32, bitcast=True)
    y = tl.full((), DENOMINATOR, tl.float32)
    if INLINE:
        result = tl.inline_asm_elementwise("div.full.f32 $0, $1, $2;", "=f,f,f", [x, y], dtype=tl.float32,
                                           is_pure=True, pack=1)
    else:
        result = x / y
    tl.store(Z + offsets, result.to(tl.int32, bitcast=True))


def _require_tileir():
    if type(driver.active).__name__ != "TileIRDriver":
        pytest.skip("requires the TileIR backend")


def _special_bits():
    return np.array([
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF,
        0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001
    ], dtype=np.uint32)


def _to_device(bits):
    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int32)).cuda()


def _bits(tensor):
    torch.cuda.synchronize()
    return tensor.cpu().numpy().view(np.uint32)


def test_div_full_inline_asm_matches_native_division():
    _require_tileir()
    rng = np.random.RandomState(0)
    # Sample the full FP32 encoding range, including subnormals and NaN payloads.
    x_bits = rng.randint(0, 2**32, size=2**16, dtype=np.uint32)
    y_bits = rng.randint(0, 2**32, size=2**16, dtype=np.uint32)
    special = _special_bits()
    # Every special numerator against every special denominator.
    x_bits[:special.size**2] = np.repeat(special, special.size)
    y_bits[:special.size**2] = np.tile(special, special.size)
    x, y = _to_device(x_bits), _to_device(y_bits)
    inline, native = torch.empty_like(x), torch.empty_like(x)
    grid = (x_bits.size // BLOCK, )
    _divide[grid](x, y, inline, True, BLOCK=BLOCK)
    _divide[grid](x, y, native, False, BLOCK=BLOCK)
    np.testing.assert_array_equal(_bits(inline), _bits(native))


def test_div_full_inline_asm_accuracy():
    _require_tileir()
    rng = np.random.RandomState(0)
    # Keep inputs and results normal: the instruction is specified to 2 ULP.
    x = rng.uniform(0.5, 2, BLOCK).astype(np.float32)
    x[::2] *= -1
    y = rng.uniform(0.5, 2, BLOCK).astype(np.float32)
    y = np.ldexp(y, np.linspace(-125, 124, y.size).astype(np.int32))
    y[0], y[-1] = 2.0**-126, 2.0**126
    x[-1] = 1.0
    y[::3] *= -1
    x_tri, y_tri = _to_device(x.view(np.uint32)), _to_device(y.view(np.uint32))
    out = torch.empty_like(x_tri)
    _divide[(1, )](x_tri, y_tri, out, True, BLOCK=BLOCK)
    np.testing.assert_array_max_ulp(_bits(out).view(np.float32), x / y, maxulp=2)


@pytest.mark.parametrize("denominator_bits", [
    0x40400000, 0xC0400000, 0x00800000, 0x80800000, 0x7E800000, 0xFE800000, 0x3DCCCCCD, 0xC2F6E979
])
def test_div_full_inline_asm_matches_constant_division(denominator_bits):
    _require_tileir()
    rng = np.random.RandomState(0)
    x_bits = rng.randint(0, 2**32, size=2**14, dtype=np.uint32)
    special = _special_bits()
    x_bits[:special.size] = special
    denominator = np.array([denominator_bits], dtype=np.uint32).view(np.float32).item()
    x = _to_device(x_bits)
    inline, native = torch.empty_like(x), torch.empty_like(x)
    grid = (x_bits.size // BLOCK, )
    _divide_by_constant[grid](x, inline, denominator, True, BLOCK=BLOCK)
    _divide_by_constant[grid](x, native, denominator, False, BLOCK=BLOCK)
    np.testing.assert_array_equal(_bits(inline), _bits(native), err_msg=f"denominator bits: 0x{denominator_bits:08x}")
