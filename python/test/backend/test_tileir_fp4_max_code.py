"""Packed maximum of FP4 magnitude codes, as used by triton_kernels."""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _max_code(src, dst, N: tl.constexpr):
    offsets = tl.arange(0, N)
    packed = tl.load(src + offsets).to(tl.uint32)
    out = tl.inline_asm_elementwise(
        """{
        .reg .b32 a, b;
        and.b32 a, $1, 0x00070007;
        shr.u32 b, $1, 4;  and.b32 b, b, 0x00070007; max.u16x2 a, a, b;
        shr.u32 b, $1, 8;  and.b32 b, b, 0x00070007; max.u16x2 a, a, b;
        shr.u32 b, $1, 12; and.b32 b, b, 0x00070007; max.u16x2 a, a, b;
        shr.u32 b, a, 16;
        and.b32 a, a, 7;
        max.u32 $0, a, b;
        }""",
        constraints="=r,r",
        args=[packed],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    tl.store(dst + offsets, out)


def test_packed_fp4_max_code():
    rng = np.random.default_rng(20261110)
    fixed = [0x00000000, 0xFFFFFFFF, 0x88888888, 0x77777777, 0x80000000, 0x70000000, 0x00000007, 0x12345678]
    # One code per nibble position, with the sign bit of every nibble set.
    single = [0x88888888 | (code << (4 * k)) for k in range(8) for code in range(8)]
    values = np.array(fixed + single, dtype=np.uint32)
    values = np.concatenate([values, rng.integers(0, 2**32, size=256 - values.size, dtype=np.uint32)])
    assert values.size == 256
    expected = np.zeros_like(values)
    for k in range(8):
        expected = np.maximum(expected, (values >> np.uint32(4 * k)) & np.uint32(7))

    src = torch.from_numpy(values.view(np.int32)).cuda()
    dst = torch.empty_like(src)
    _max_code[(1, )](src, dst, values.size)
    np.testing.assert_array_equal(dst.cpu().numpy().view(np.uint32), expected)
