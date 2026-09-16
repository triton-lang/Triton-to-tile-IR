See [The original Triton README](https://github.com/triton-lang/Triton-to-tile-IR/blob/main/README.original.md) for more details.

## ⚡ Helion Hackathon — Performance Tuning Guide

**The default backend is OSS PTX. The linked Triton 3.6 wheel is a previous release; this source branch uses Triton 3.7 and CUDA Tile IR 13.4. Using [Helion](https://github.com/pytorch/helion) with the TileIR backend([whl](https://github.com/triton-lang/Triton-to-tile-IR/releases/download/v3.6.0-rc1/nvtriton-3.6.0-cp313-cp313-linux_x86_64.whl))?** Check out the **[Helion TileIR Backend Performance Tuning Guide](HelionPerformanceTuningGuide.md)** for config recipes, autotuning strategies, and porting tips.

### ⚠️ How to Submit TileIR Result

> **You MUST set both `ENABLE_TILE=1` and `HELION_BACKEND=tileir` via `os.environ` at the top of your `submission.py`, before any `import helion` or `import triton` statements.** These environment variables must be set before the modules are imported to take effect.

```python
import os
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"

# Now import helion/triton — they will pick up the TileIR backend
import helion
import helion.language as hl

# ... rest of your submission code ...
```

### Troubleshooting: Porting Configs from Triton Backend

> **Do NOT directly reuse Helion configs tuned for the Triton (PTX) backend.** The two backends have different tuning knobs with different semantics — directly porting configs will likely result in poor performance or tuning errors like below

| Error | Cause | Fix |
|-------|-------|-----|
| `InvalidConfig: Too many values for config['range_unroll_factors']` | tileir doesn't support `range_*` params | Remove `range_flattens`, `range_multi_buffers`, `range_num_stages`, `range_unroll_factors`, `range_warp_specializes`, `static_ranges` |
| `InvalidConfig: Too many values for config['static_ranges']` | Same as above | Same — remove all `range_*` and `static_ranges` |

- **Remove unsupported knobs**: TileIR does not support `range_unroll_factors`, `range_multi_buffers`, `range_flattens`, `range_warp_specialize`, `load_eviction_policies`, `static_ranges`, `indexing="block_ptr"`, etc. See the full list in the [Helion TileIR Performance Tuning Guide](HelionPerformanceTuningGuide.md#knobs-not-available-on-tileir). If you want to try the TileIR backend with default-backend-tuned config, remember to remove the unsupported configs.
- **Recommended**: Start autotuning from scratch. The TileIR backend has its own set of knobs (`occupancy`, `num_ctas`, wider `num_stages` range) and the autotuner will explore them effectively.

**Triton-TileIR backend general optimization tips?** See the **[Performance Tuning Tips](third_party/tileir/PerformanceTuningTips.md)** for occupancy, num_ctas, TMA API preferences, num_stages tuning, and benchmark results.

---

# Triton CUDA Tile IR Backend
This incubator repo adds the CUDA Tile IR backend to Triton. Users can enable the CUDA Tile IR backend by setting the environment variable `ENABLE_TILE=1`. This branch uses Triton 3.7, public CUDA Tile IR v13.4.0 sources, and matching 13.4.59 compiler and runtime components. Set `ENABLE_TILE=1` before importing Triton.

## How to install?
doesn't change
```
pip install -e .
```

## How to run CUDA Tile IR Backend?

```bash
export ENABLE_TILE=1
```

## Known functional issues

CUDA Tile IR now supports only an unordered memory model, where global memory access operations are not ordered by default. If explicit memory access ordering is required, memory token semantics are available for users to control this behavior.
Currently, the implementation includes only APIs that are compatible with existing Triton APIs for current Triton kernels. Support for memory tokens will require extending the Triton APIs. We plan to submit another MR to extend Triton APIs for the CUDA Tile memory model later.
At this stage, the following workloads may produce incorrect results unless the script is updated:

- When there is memory aliasing between different global memory access operations.
- When data transactions occur across different tile blocks (e.g., splitK/streamK), where deterministic reduction across tile blocks requires lock logic in global memory.

Potential future solutions (to be discussed):

- Extend Triton APIs to explicitly support the unordered memory model (scripts will need revision).
- Abstract global memory locks into an independent API.
- Apply conservative rules to append memory tokens during Triton-to-CUDA Tile conversion, which avoids script changes but may introduce performance loss.

**CUDA 13.4 status:** The memory-model discussion above is retained as design context. The conservative token-insertion approach is now implemented for supported Triton memory operations, including aliasing accesses and descriptor reductions. Direct user control of memory tokens remains a separate API concern, and ordering within one tile block does not replace the cross-block synchronization required by splitK/streamK and lock-based protocols.

## Known performance issues
- Small GEMM performance is currently poor (will be addressed in a future CUDA release).
- Kernels using legacy tensor-of-pointer load/store APIs exhibit poor performance (will be addressed in a future CUDA release).
- `num_warps` is accepted for compatibility but does not directly control TileIR warp allocation. For XXXNorm kernels with large reduction dimensions, performance may degrade due to register spilling (support may be added in a future CUDA release).
- Ordinary `tl.gather` on large tiles can be significantly slower than the NVIDIA PTX backend. Benchmark performance-sensitive uses.

## Performance Tuning Tips
- New hints for CUDA Tile IR backend: `occupancy` (critical). The occupancy hint accepts an integer N from 1 to 32, indicating that the programmer expects N active thread blocks to run simultaneously per SM. This hint is 1 by default and is worth tuning for many compute-intensive kernels.
- Existing Triton hints: `num_ctas` (critical). Setting `num_ctas=2` is critical for dense dot-related workloads, as it enables 2CTA mode MMA on Blackwell architecture.
- For guidance on performance tuning, please refer to the detailed tips provided [here](third_party/tileir/PerformanceTuningTips.md).

## ChangeList
### Triton’s core files changes:

1. When `ENABLE_TILE=1` is set, the default CUDA target is switched to the CUDA Tile IR target. Changes are made to `driver.py` and `compiler.py`.
2. When a compilation bug occurs with the CUDA Tile IR Backend, it can fall back to the NVIDIA PTX backend. Main changes include `jit.py` and `nvidia/backend/driver.py`. In this release, automatic fallback is disabled by default; set `TRITON_TILEIR_RUNTIME_FALLBACK=1` to enable it.
3. Support for lowering Triton host TMA APIs to CUDA Tile IR's TMA APIs. Triton provides both host and device TMA implementations, but CUDA TileIR only has the device implementation (internally, the CUDA Tile IR compiler determines whether to use host or device; however, in the language, only the kernel-level API exists). Main files modified: `core.py`, `semantic.py`, `tensor_descriptor.py`.
4. CUDA Tile IR disables approx by default. To enable approx, pls use `export TILEIR_ENABLE_APPROX=1`
5. CUDA Tile IR disables FTZ by default. To enable FTZ , pls use `export TILEIR_ENABLE_FTZ=1`

### CUDA Tile IR Backend support:

1. Conversion pass: converts TTIR to CUDA Tile IR. Implemented in `TritonToCudaTile.*`
2. Rewrite assume pass: converts assume ops in TTIR/LLVM IR to CUDA Tile IR assume ops. Implemented in `rewriteAssume.*`
3. Python code: mostly aligned with `third_party/nvidia/backend`.

## CUDA Tile IR in CUDA 13.4
This release is validated on Blackwell (B200). Availability of individual operations also depends on the GPU architecture.
### Dependency
Triton CUDA Tile IR backend depends on `bin/tileiras`, `bin/ptxas`, and `nvvm/lib64/libnvvm.so` from CUDA 13.4 (13.4.59).
Triton CUDA Tile IR backend also depends on the [CUDA Tile IR dialect](https://github.com/NVIDIA/cuda-tile).

### Auto Tune
CUDA Tile IR accepts `num_warps` for compatibility, while `occupancy` controls the expected number of active thread blocks per SM. **In practice, we have found that `occupancy` and `num_ctas` are crucial to CUDA Tile IR performance.**

### Supported operations and features
- Tensor descriptor (TMA) gather/scatter, atomic reductions, and load padding.
- Ordinary `tt.gather`, `tt.unsplat`, and `tt.map_elementwise`.
- Matching FP4/FP8 scaled matrix multiplication, including single-scale FP8.
- More native math and conversion operations, bf16 atomic add, source-line information, and memory ordering improvements.

### Operations and features not yet supported or fully supported:
- `tt.elementwise_inline_asm` (general inline PTX; only selected forms have native lowerings)
- `cf.cond_br` paths that cannot be converted to supported structured control flow
- `cuda_tile.reduce` (only pure operations allowed)
- `tt.dot_scaled` (mixed operand types, single-scale FP4, and missing scales on both operands)
- `cuda_tile.ftof` (f32-to-FP8 E5M2 conversion with rtz rounding)
- `tt.extern_elementwise` (only mapped libdevice functions are supported)
- TMA reduce (floating-point min/max are not supported; floating-point add and supported integer reductions are available)
- `math.erf`
- `atomic_cas` (bf16 and fp16 not supported)
- TMA read-modify-write operations that return the previous values (descriptor reductions do not return values)
- TMA offsets must satisfy the contiguous dimension's 16-byte alignment requirement
- i64 tensor descriptor coordinates (use i32 coordinates; this does not restrict ordinary pointer indexing or descriptor strides to i32)
- Conditional replacement of tensor descriptors
- Block pointers and histogram
- Gluon, Proton instrumentation, and PTX/TTGIR/LLIR inspection or override
