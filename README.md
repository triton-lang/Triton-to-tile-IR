See [The original Triton README](https://github.com/triton-lang/Triton-to-tile-IR/blob/main/README.original.md) for more details.

# Triton CUDA Tile IR Backend
This incubator repo adds the CUDA Tile IR backend to Triton. Users can enable the CUDA Tile IR backend by setting the environment variable `ENABLE_TILE=1`. The CUDA Tile IR backend in this repo only uses features available in CUDA 13.1.

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

## Known performance issues
- Small GEMM performance is currently poor (will be addressed in a future CUDA release).
- Kernels using legacy tensor-of-pointer load/store APIs exhibit poor performance (will be addressed in a future CUDA release).
- `num_warps` is not exposed yet. For XXXNorm kernels with large reduction dimensions, performance may degrade due to register spilling (support may be added in a future CUDA release).

## Performance Tuning Tips
- New hints for CUDA Tile IR backend: `occupancy` (critical). The occupancy hint accepts an integer N from 1 to 32, indicating that the programmer expects N active thread blocks to run simultaneously per SM. This hint is 1 by default and is worth tuning for many compute-intensive kernels.
- Existing Triton hints: `num_ctas` (critical). Setting `num_ctas=2` is critical for dense dot-related workloads, as it enables 2CTA mode MMA on Blackwell architecture.
- For guidance on performance tuning, please refer to the detailed tips provided [here](third_party/tileir/PerformanceTuningTips.md).

## ChangeList
### Triton’s core files changes:

1. When `ENABLE_TILE=1` is set, the default CUDA target is switched to the CUDA Tile IR target. Changes are made to `driver.py` and `compiler.py`.
2. When a compilation bug occurs with the CUDA Tile IR Backend, it falls back to the NVIDIA PTX backend. Main changes include `jit.py` and `nvidia/backend/driver.py`.
3. Support for lowering Triton host TMA APIs to CUDA Tile IR's TMA APIs. Triton provides both host and device TMA implementations, but CUDA TileIR only has the device implementation (internally, the CUDA Tile IR compiler determines whether to use host or device; however, in the language, only the kernel-level API exists). Main files modified: `core.py`, `semantic.py`, `tensor_descriptor.py`.
4. CUDA Tile IR disables approx by default. To enable approx, pls use `export TILEIR_ENABLE_APPROX=1`
5. CUDA Tile IR disables FTZ by default. To enable FTZ , pls use `export TILEIR_ENABLE_FTZ=1`

### CUDA Tile IR Backend support:

1. Conversion pass: converts TTIR to CUDA Tile IR. Implemented in `TritonToCudaTile.*`
2. Rewrite assume pass: converts assume ops in TTIR/LLVM IR to CUDA Tile IR assume ops. Implemented in `rewriteAssume.*`
3. Python code: mostly aligned with `third_party/nvidia/backend`.
 
## CUDA Tile IR in CUDA 13.1
We only support Blackwell GPU in CUDA 13.1.
### Dependency
Triton CUDA Tile IR backend depends on `bin/tileiras`, `bin/ptxas`, and `nvvm/lib64/libnvvm.so` from CUDA 13.1.
Triton CUDA Tile IR backend also depends on the [CUDA Tile IR dialect](https://github.com/NVIDIA/cuda-tile).

### Auto Tune
CUDA Tile IR in CUDA 13.1 doesn't support `num_warps` (but may support it in a future CUDA release), while CUDA Tile IR adds a new tuning attribute `occupancy`. **In practice, we have found that `occupancy` and `num_ctas` are crucial to CUDA Tile IR performance.**

### Operations and features not yet supported or fully supported:
- `tt.elementwise_inline_asm`
- `cf.cond_br`
- `cuda_tile.reduce` (only pure operations allowed)
- `tt.gather`
- `tt.unsplat`
- `tt.dot_scaled`
- `cuda_tile.ftof` (rtz mode not supported)
- `tt.extern_elementwise`
- `tt.map_elementwise`
- TMA scatter feature
- TMA gather feature
- TMA reduce feature
- TMA load padding default value
- `math.erf`
- `atomic_rmw` (bf16 dtype not supported)
- `atomic_cas` (bf16 and fp16 not supported)
- TMA rmw feature
- TMA arbitrary offset is not supported yet
- i64 index type of the memref is not supported yet

