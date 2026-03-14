# Helion Performance Tuning Guide for CUDA TileIR Backend

This guide helps [Helion](https://github.com/pytorch/helion) users optimize kernel performance when targeting the CUDA TileIR backend on Blackwell GPUs (sm_100+).

For raw Triton kernel tuning tips, see [PerformanceTuningTips.md](third_party/tileir/PerformanceTuningTips.md).

## Environment Setup

Before any Python import, set:

```bash
export ENABLE_TILE=1
export HELION_BACKEND=tileir
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
- **2**: Enables 2CTA mode MMA on Blackwell. Beneficial when GEMM tiles are wide (BM × BN ≥ 256 × 128). Can cause accuracy issues with small block sizes — verify correctness.

### 4. `num_stages` — Pipeline Depth as Cost Hint

Unlike Triton's PTX backend, TileIR treats `num_stages` as a **cost hint** rather than a strict directive. The tileiras compiler determines the optimal pipeline configuration from a whole-program perspective.

- Range: {1..10} (wider than Triton's 1-4)
- **Expand the tuning range** during autotune, especially for dot-related kernels
- Higher values can be beneficial — don't limit to 3-4

### 5. `block_sizes` — Tile Dimensions

- All values must be powers of 2
- Batch dimension → always 1 (Helion enforces for 3D operands)
- TileIR treats tiles as **CGA-level representations** — consider larger block sizes than you would use on PTX

### 6. Numerical Precision Options

TileIR disables `approx` and `ftz` by default (unlike the Triton PTX backend):

```bash
export TILEIR_ENABLE_APPROX=1   # Enable approximate math
export TILEIR_ENABLE_FTZ=1      # Enable flush-to-zero
```

These can improve performance for **attention** and variant kernels with acceptable precision trade-offs.

> **Note**: The tileiras compiler in CUDA 13.1 does not automatically optimize `exp.approx → ex2 + mulf`. For performance parity with PTX, explicitly rewrite `expOp` to use `ex2 + mulf`.

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
    num_stages=2,
    num_ctas=1,
    occupancy=4,               # memory-bound
    indexing="pointer",
)
```

### LayerNorm / RMSNorm

```python
helion.Config(
    block_sizes=[4, 1024],     # tokens, hidden
    num_stages=2,
    num_ctas=1,
    occupancy=4,
    indexing="pointer",
)
```

## Autotuning

### Quick Validation (no autotuning)

```python
@helion.kernel(config=helion.Config(...))
def my_kernel(...): ...
```

### Seed + Autotune (recommended)

Provide a known-good config as seed, let the autotuner explore neighbors:

```python
@helion.kernel(
    configs=[helion.Config(
        block_sizes=[128, 128, 32],
        indexing="tensor_descriptor",
        num_stages=4, num_ctas=1, occupancy=2,
    )],
    autotune_effort="medium",   # ~100 configs
)
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

| Level | Configs | Use Case |
|-------|---------|----------|
| `"none"` | 0 | Use provided config only |
| `"low"` | ~20 | Quick exploration |
| `"medium"` | ~100 | Good balance |
| `"high"` | ~500+ | Thorough search |

The autotuner searches: `block_sizes`, `num_stages` {1..10}, `num_ctas` {1,2}, `occupancy` {1,2,4,8}, `indexing`, `pid_type`, `l2_groupings`.

## Porting Checklist: Triton Backend → TileIR

1. Set environment: `ENABLE_TILE=1`, `HELION_BACKEND=tileir`
2. Replace `indexing="block_ptr"` → `"tensor_descriptor"` (for dot kernels) or `"pointer"`
3. Set `num_warps=4` (or remove — defaults to 4)
4. Add TileIR knobs: `num_ctas=1`, `occupancy=2` as starting point
5. Remove unsupported: `range_unroll_factors`, `range_multi_buffers`, `range_flattens`, `range_warp_specialize`, `load_eviction_policies`, `static_ranges`
6. Widen `num_stages` range: TileIR supports 1-10 (vs Triton 1-4)
7. Test correctness: compare output against `torch` reference
8. Benchmark: compare vs cuDNN / Triton backend baseline

```python
# Before (Triton backend):
helion.Config(block_sizes=[128,128,32], num_warps=8, indexing="block_ptr",
              num_stages=3, range_unroll_factors=[2], load_eviction_policies=["last"])

# After (TileIR backend):
helion.Config(block_sizes=[128,128,32], num_warps=4, indexing="tensor_descriptor",
              num_stages=4, num_ctas=1, occupancy=2)
```

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
| `block_ptr` silently stripped | TileIR doesn't support block_ptr | Use `"tensor_descriptor"` or `"pointer"` |
| Autotuner NaN / accuracy mismatch | Some config combos produce wrong results | Autotuner filters automatically; verify manual configs |
| Compile timeout | Large configs take >60s via tileiras | Reduce `block_sizes` or `num_stages` |
| Stale cached kernels | Triton cache not cleared after config change | `rm -rf ~/.triton/cache` |

## References

- [Triton TileIR Backend Performance Tuning Tips](third_party/tileir/PerformanceTuningTips.md) — raw Triton kernel tuning
- [Helion PR #1250](https://github.com/pytorch/helion/pull/1250) — TileIR backend autotuner integration
- [CUDA Tile IR dialect](https://github.com/NVIDIA/cuda-tile) — upstream TileIR dialect
- [Install Guide](INSTALL.md) — how to install the Triton TileIR backend
