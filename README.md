See [The original Triton README](https://github.com/triton-lang/Triton-to-tile-IR/blob/main/README.original.md) for more details.

# Triton CUDA TileIR Backend
This incubator repo adds the CUDA TileIR backend to Triton. Users can enable the CUDA TileIR backend by setting the environment variable ENABLE_TILE=1. CUDA TileIR backend in this repo only contains CUDA 13.1's features.

## How to install?
doesn't change
```
pip install -e .
```
## How to run CUDA TileIR backend?
```
export ENABLE_TILE=1
```

## Known functional issues:
CUDA TileIR now only supports an unordered memory model, where the global memory access ops by default are not ensured with an access ordering. When the user requires for an explicit memory access ordering, memory token semantics are provided for users to control.
However, current implementation only contains the APIs compatible with current Triton APIs for existing triton kernels. The support of the memory token will need to extend the current triton APIs. We plan to file another MR (extending Triton APIs to support cuda_tile memory model) later.

At this stage, the following workloads may result in wrong results if not changing the script:
- when there is memory aliasing between different global memory access Ops.
- when there are data transaction across different tile blocks, e.g. splitK/streamK where you'll need an deterministic reduction across tile blocks and needs a lock logic in the gmem.

Potential ways to resolve this in the future (to be discussed later)
- extend Triton API to explicitly support the unordered memory model (We'll need to revise the script then). Or,
- abstract the gmem locks into independent API, or,
- some conservative rules to append men_tokens during triton->cuda_tile, which doesn't need to change the script but may introduce perf loss. Or,


## Known performance issues:
- small gemm perf is bad (will be fixed in future CUDA version)
- kernels written in the legacy tensor-of-ptr load/store APIs have bad perf (will be fixed in future CUDA version) 
- num_warps is not exposed for now, XXXNorm kernels in case of large reduction dim size may result from poor perf from reg spilling (may support it in future CUDA version).

## Perf Tips
- New Hints for CUDA TileIR Backend: `occupancy` (Critical). The occupancy hint accepts an integer N from 1 to 10, indicating that the programmer expects N thread blocks to run simultaneously on a single SM. This hint is 1 by default and is worth tuning for many compute-intensive kernels.
- Existing Triton Hints: `num_ctas` (Critical). Setting num_ctas=2 is critical for dense dot-related workloads, as it enables 2CTA mode MMA on Blackwell architecture.


## ChangeList
### Triton’s core files changes:

1. When `ENABLE_TILE=1`, the default CUDA target is switched to the CUDA TileIR target. Changes are made to `driver.py` and `compiler.py`.
2. When a compilation bug occurs with the CUDA TileIR backend, it falls back to the NVIDIA PTX backend. Main changes include `jit.py` and `nvidia/backend/driver.py`.
3. Support for lowering Triton host TMA APIs to CUDA TileIR's TMA APIs. Triton provides both host and device TMA implementations, but CUDA TileIR only has the device implementation (internally, the CUDA TileIR compiler determines whether to use host or device; however, in the language, only the kernel-level API exists). Main files modified: `core.py`, `semantic.py`, `tensor_descriptor.py`.
4. CUDA TileIR disable approx by default. To enable approx, pls use `export TILEIR_ENABLE_APPROX=1`
5. CUDA TileIR disable FTZ by default. To enable FTZ , pls use `export TILEIR_ENABLE_FTZ=1`

### CUDA TileIR backend support:

1. Conversion pass: converts TTIR to CUDA Tile IR. Implemented in `TritonToCudaTile.*`
2. Rewrite assume pass: converts assume ops in TTIR/LLVM IR to CUDA Tile IR assume ops. Implemented in `rewriteAssume.*`
3. Python code: mostly aligned with `third_party/nvidia/backend`.
 
## CUDA TileIR in CUDA 13.1
We only support Blackwell GPU in CUDA 13.1.
### Dependency
Triton CUDA TileIR backend depends on bin/tileiras, bin/ptxas and nvvm/lib64/libnvvm.so from CUDA 13.1.
Triton CUDA TileIR backend also depends on CUDA TileIR dialect (https://github.com/NVIDIA/cuda-tile).

### Auto Tune
CUDA TileIR in CUDA 13.1 doesn't support num_warp (but may support it in future CUDA), while CUDA TileIR adds a new tuning attribute "occupancy".  **In practice, we have found that "occupancy" and "num_ctas" are crucial to CUDA TileIR perf.**

### Operations and features not yet supported or fully supported:
- tt.elementwise_inline_asm
- cf.cond_br
- cuda_tile.reduce (only pure operations allowed)
- tt.gather
- tt.unsplat
- tt.dot_scaled
- cuda_tile.ftof (rtz mode not supported)
- tt.extern_elementwise
- tt.map_elementwise
- tma scatter feature
- tma gather feature
- tma reduce feature
- tma load padding default value
- math.erf
- atomic_rmw (bf16 dtype not supported)
- atomic_cas (bf16 and fp16 not supported)
- tma rmw feature
- TMA arbitrary offset hasn't been supported yet
- i64 index type of the memref hasn't been support yet

