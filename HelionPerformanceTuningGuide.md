# Helion Performance Tuning Guide for CUDA TileIR Backend

This guide helps [Helion](https://github.com/pytorch/helion) users optimize kernel performance when targeting the CUDA TileIR backend on Blackwell GPUs (sm_100+).

For raw Triton kernel tuning tips, see [PerformanceTuningTips.md](third_party/tileir/PerformanceTuningTips.md).

## Environment Setup

Set both environment variables via `os.environ` **at the top of your script, before any `import helion` or `import triton`**:

```python
import os
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"

# Optional: precision trade-offs for attention/softmax kernels
os.environ["TILEIR_ENABLE_APPROX"] = "1"
os.environ["TILEIR_ENABLE_FTZ"] = "1"

# Now import helion/triton — they will pick up the TileIR backend
import helion
import helion.language as hl
```

Verify:
```python
from helion._compat import use_tileir_tunables
assert use_tileir_tunables(), "TileIR not active"
```

## TileIR Config Knobs

### Available Knobs

| Knob | Type | Range | Default | Effect |
|------|------|-------|---------|--------|
| `block_sizes` | list[int] | powers of 2 | auto | Tile dimensions — **most important knob** |
| `indexing` | str | `"pointer"`, `"tensor_descriptor"` | auto | Memory access pattern — TMA vs pointer arithmetic |
| `num_stages` | int | {1..10} | auto | Pipeline depth (cost hint, not strict directive) |
| `num_ctas` | int | {1, 2} | 1 | CTAs per CGA — enables DSMEM sharing between CTAs |
| `occupancy` | int | 1-32 (autotuner searches {1, 2, 4, 8}) | 1 | SM utilization target — register pressure vs latency hiding |
| `num_warps` | int | 4 (fixed) | 4 | **Not tunable** on TileIR |

### Knobs NOT Available on TileIR

Remove these when porting from the Triton backend:

`indexing="block_ptr"`, `range_unroll_factors`, `range_multi_buffers`, `range_flattens`, `range_warp_specialize`, `load_eviction_policies`, `static_ranges`

## Key Tuning Principles

### 1. `indexing` — Choose the Right Memory Access Pattern

- **`"tensor_descriptor"`** — Use for any kernel with `hl.dot` / `torch.bmm` / `torch.addmm` / `torch.matmul`. Maps to TMA (Tensor Memory Accelerator) hardware loads. Critical for GEMM/attention performance.
- **`"pointer"`** — Use for elementwise, reductions, and norms. Simple pointer arithmetic with lower overhead.
- **Never** use `"block_ptr"` — not supported on TileIR.

### 2. `occupancy` — Balance Compute vs Memory

- **1-2**: More registers per thread. Best for **compute-bound** kernels (GEMM, matmul, attention).
- **4-8**: More concurrent warps. Best for **memory-bound** kernels (elementwise, reduction, norm).
- Manual configs can go up to **32**. The autotuner searches {1, 2, 4, 8} by default.
- Heuristic: kernel has `hl.dot` → start with 1-2. No `hl.dot` → start with 4.

### 3. `num_ctas` — Enable 2CTA Mode for Wide GEMMs

- **1** (default): Always safe.
- **2**: Enables 2CTA mode MMA on Blackwell. Beneficial when GEMM tiles are wide (BM × BN ≥ 256 × 128).

### 4. `num_stages` — Pipeline Depth as Cost Hint

Unlike Triton's PTX backend, TileIR treats `num_stages` as a **cost hint** rather than a strict directive. The tileiras compiler determines the optimal pipeline configuration from a whole-program perspective.

- Range: {1..10} (wider than Triton's 1-4)
- **Expand the tuning range** during autotune, especially for dot-related kernels
- Higher values can be beneficial — don't limit to 3-4

### 5. `block_sizes` — Tile Dimensions

- All values must be powers of 2
- Batch dimension → always 1 (Helion enforces for 3D operands)
- TileIR treats tiles as **CGA-level representations** — consider larger block sizes than you would use on PTX

## Per-Kernel Config Recipes

### Elementwise (add, mul, activation, cast)

Pattern: Single `hl.tile(x.shape)` loop, no reduction, no dot.

```python
helion.Config(
    block_sizes=[128, 128],
    num_stages=3,
    num_ctas=1,
    occupancy=4,          # memory-bound → high occupancy
    indexing="pointer",
)
```

### GEMM / Matmul

Pattern: Two outer tile loops + inner reduction loop with `hl.dot`.

```python
# Standard GEMM
helion.Config(
    block_sizes=[128, 128, 32],
    indexing="tensor_descriptor",  # TMA — critical for GEMM
    num_stages=4,
    num_ctas=1,
    occupancy=2,
)

# Large GEMM (M, N >= 4096)
helion.Config(
    block_sizes=[128, 256, 64],
    indexing="tensor_descriptor",
    num_stages=4,
    num_ctas=2,                    # 2CTA mode for wide tiles
    occupancy=2,
)
```

Tuning priorities: `indexing` (must be `tensor_descriptor`), `block_sizes` (BK=32-64), `num_stages` (4-8), `num_ctas` (try 2 for large M,N).

### Flash Attention

Pattern: Batch+seq outer loops, KV inner reduction with online softmax.

```python
@helion.kernel(
    static_shapes=True,
    config=helion.Config(
        block_sizes=[1, 128, 128],       # batch=1, M_tile, KV_tile
        indexing="tensor_descriptor",
        num_stages=3,
        num_ctas=1,
        occupancy=2,
    ),
)
```

Tuning priorities: `block_sizes[1]` (M_tile: 64-256), `block_sizes[2]` (KV_tile: match or half head_dim), `num_stages` (2-4), `occupancy` (1-2 for register-heavy).

### Softmax / Row Reduction

```python
helion.Config(
    block_sizes=[128, 1024],   # rows, cols — cover full row if possible
    num_stages=2,              # prefer a smaller value
    num_ctas=1,
    occupancy=4,               # memory-bound， set to 4 or larger value
    indexing="pointer",
)
```

### LayerNorm / RMSNorm

```python
helion.Config(
    block_sizes=[4, 1024],     # tokens, hidden
    num_stages=2,              # prefer a smaller value
    num_ctas=1,
    occupancy=4,               # memory-bound， set to 4 or larger value
    indexing="pointer",
)
```

## Autotuning

### Quick Validation (no autotuning)

```python
@helion.kernel(config=helion.Config(...))
def my_kernel(...): ...
```

### Manual Config List

```python
configs = [
    helion.Config(block_sizes=[64, 64, 32], num_stages=3, num_ctas=1, occupancy=1, indexing="tensor_descriptor"),
    helion.Config(block_sizes=[128, 128, 32], num_stages=4, num_ctas=1, occupancy=2, indexing="tensor_descriptor"),
    helion.Config(block_sizes=[128, 128, 64], num_stages=4, num_ctas=2, occupancy=2, indexing="tensor_descriptor"),
    helion.Config(block_sizes=[128, 256, 64], num_stages=6, num_ctas=2, occupancy=4, indexing="tensor_descriptor"),
]

@helion.kernel(configs=configs)
def my_kernel(...): ...
```

### Effort Levels

| Level | Typical duration | Use Case |
|-------|---------|----------|
| `"none"` | Near zero | Use provided config only |
| `"quick"` | Minutes | Quick exploration |
| `"full"` | Tens of minutes | Thorough search |

The autotuner searches: `block_sizes`, `num_stages` {1..10}, `num_ctas` {1,2}, `occupancy` {1,2,4,8}, `indexing`, `pid_type`, `l2_groupings`...

### Custom CUDA Graph Benchmark Function for Low-Latency Kernels

Helion supports specifying a custom CUDA graph-based benchmark function via `autotune_benchmark_fn` in `@helion.kernel`. This is useful for low-latency kernels where more precise timing can better guide the autotuner's search. **Trade-off**: the benchmarking itself takes longer due to CUDA graph capture and cache clearing overhead.

The following `do_bench_cudagraph_with_cache_clear` function uses CUDA graphs to eliminate launch overhead and explicitly clears the L2 cache between runs:

```python
from typing import Callable
import torch
import triton

def do_bench_cudagraph_with_cache_clear(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
) -> list[float]:
    """
    Benchmark with CUDA graphs and explicit L2 cache clearing.
    Returns mean execution time in milliseconds per function.
    """
    ret = []
    for fn in fns:
        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
        clear_cache_fn = cache.zero_

        with torch.cuda.stream(torch.cuda.Stream()):
            # Warmup
            clear_cache_fn()
            fn()

            # Estimate execution time
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(5):
                clear_cache_fn()
                fn()
            end_event.record()
            torch.cuda.synchronize()
            estimate_ms = start_event.elapsed_time(end_event) / 5

            n_repeat = 1000 if estimate_ms == 0 else max(1, int(repeat / estimate_ms))

            # CUDA graph: cache clear + kernel
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(n_repeat):
                    clear_cache_fn()
                    fn()
            torch.cuda.synchronize()

            # CUDA graph: cache clear only (to subtract overhead)
            cache_clear_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cache_clear_graph):
                for _ in range(n_repeat):
                    clear_cache_fn()
            torch.cuda.synchronize()

            # Measure cache clear time
            cache_clear_start = torch.cuda.Event(enable_timing=True)
            cache_clear_end = torch.cuda.Event(enable_timing=True)
            cache_clear_start.record()
            cache_clear_graph.replay()
            cache_clear_end.record()
            torch.cuda.synchronize()
            cache_clear_time = cache_clear_start.elapsed_time(cache_clear_end) / n_repeat

            # Measure total time
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            g.replay()
            end_event.record()
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event) / n_repeat

        # Pure kernel time = total - cache clear overhead
        ret.append(total_time - cache_clear_time)
    return ret
```

Usage with `@helion.kernel`:

```python
@helion.kernel(
    ...
    autotune_benchmark_fn=do_bench_cudagraph_with_cache_clear,
)
def my_low_latency_kernel(...):
    ...
```

> **Tip**: For more stable and reproducible benchmark results, lock the GPU clock and power limit before running autotuning. This prevents frequency throttling from introducing noise into the timing measurements.
> ```bash
> # Lock GPU clocks (example: 1980 MHz on B200)
> sudo nvidia-smi -lgc 1980,1980
> # Lock power limit (example: 1000W on B200)
> sudo nvidia-smi -pl 1000
> ```

## Porting Checklist: Triton Backend → TileIR

1. Set environment: `ENABLE_TILE=1`, `HELION_BACKEND=tileir`
2. Replace `indexing="block_ptr"` → `"tensor_descriptor"` (for dot kernels) or `"pointer"`
3. Set `num_warps=4` (or remove — defaults to 4)
4. Add TileIR knobs: `num_ctas=1`, `occupancy=1` as starting point
5. Remove unsupported: `range_unroll_factors`, `range_multi_buffers`, `range_flattens`, `range_warp_specialize`, `load_eviction_policies`, `static_ranges`
6. Widen `num_stages` range: TileIR supports 1-10 (vs Triton 1-8)
7. Test correctness: compare output against `torch` reference
8. Benchmark: compare vs Triton backend baseline or others

```python
# Before (Triton backend):
helion.Config(block_sizes=[128,128,32], num_warps=8, indexing="block_ptr",
              num_stages=3, range_unroll_factors=[2], load_eviction_policies=["last"])

# After (TileIR backend):
helion.Config(block_sizes=[128,128,32], num_warps=4, indexing="tensor_descriptor",
              num_stages=3, range_unroll_factors=[], load_eviction_policies=[""], num_ctas=1, occupancy=2)
```

## Useful Helion Environment Variables

### `HELION_AUTOTUNE_COMPILE_TIMEOUT`

Per-config compile timeout in seconds. During autotuning, bad configs (e.g. excessively large block sizes or deep pipelines) can cause the tileir backend to spend minutes on a single config. This variable sets a hard timeout to kill those slow compilations early, keeping autotune sweeps fast.

- **Default**: `60` (seconds)
- **Recommended**: `20` — aggressive enough to skip bad configs, generous enough for most valid configs

```bash
export HELION_AUTOTUNE_COMPILE_TIMEOUT=20
```

### `HELION_PRINT_OUTPUT_CODE`

Print the generated Triton IR code to stdout. Useful for inspecting what Helion produces before it gets compiled.

```bash
HELION_PRINT_OUTPUT_CODE=1 python my_script.py
```

### `TILEIR_ENABLE_APPROX` / `TILEIR_ENABLE_FTZ`

TileIR disables approximate math and flush-to-zero by default (unlike the Triton PTX backend). Enabling these can improve performance for attention and softmax kernels with acceptable precision trade-offs.

```bash
export TILEIR_ENABLE_APPROX=1   # Enable approximate math (e.g. fast exp)
export TILEIR_ENABLE_FTZ=1      # Enable flush-to-zero for denormals
```
These can improve performance for **attention** and variant kernels with acceptable precision trade-offs.

> **Note**: The tileiras compiler in CUDA 13.1 does not automatically optimize `exp.approx → ex2 + mulf`. For performance parity with PTX, explicitly rewrite `expOp` to use `ex2 + mulf`.

## Debugging

### Print Generated Triton Code
```bash
HELION_PRINT_OUTPUT_CODE=1 python my_script.py
```

### Check Backend Detection
```python
from helion._compat import use_tileir_tunables
from helion.runtime.settings import _get_backend
print(_get_backend())          # should be "tileir"
print(use_tileir_tunables())   # should be True
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `InvalidConfig: Unsupported config keys` | `HELION_BACKEND=tileir` not set | Set env vars before import |
| Autotuner NaN / accuracy mismatch | Some config combos produce wrong results | Autotuner filters automatically; verify manual configs |
| Stale cached kernels | Triton cache not cleared after config change | `rm -rf ~/.triton/cache` |

## References

- [Triton TileIR Backend Performance Tuning Tips](third_party/tileir/PerformanceTuningTips.md) — raw Triton kernel tuning
- [Helion PR #1250](https://github.com/pytorch/helion/pull/1250) — TileIR backend autotuner integration
- [CUDA Tile IR dialect](https://github.com/NVIDIA/cuda-tile) — upstream TileIR dialect
- [Install Guide](INSTALL.md) — how to install the Triton TileIR backend
