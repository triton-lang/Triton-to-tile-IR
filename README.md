# Triton CUDA Tile IR Backend

This branch combines Triton **3.7** with the public [CUDA Tile IR 13.4 sources](https://github.com/NVIDIA/cuda-tile/tree/v13.4.0). Builds bundle the matching **13.4.59** TileIR compiler and runtime components. The NVIDIA PTX backend remains the default; set `ENABLE_TILE=1` before importing Triton to use TileIR.

See the [original Triton README](README.original.md) for the frontend and language, and the [backend build guide](third_party/tileir/README.md) for installation.

## CUDA 13.4 update

- Gather/scatter through tensor descriptors, plus ordinary `tl.gather`.
- Descriptor atomic reductions: integer add/min/max/bitwise operations and floating-point add.
- FP4/FP8 scaled matrix multiplication, including single-scale FP8.
- More native math operations, FP4 conversion helpers, and TF32 support.
- Descriptor padding, warp scheduling hints, source-line information, and memory ordering improvements.

## Current limitations

- Mixed-format scaled matrix multiplication and single-scale FP4.
- Floating-point descriptor atomic min/max and conditional descriptor replacement.
- Block pointers, histogram, general inline PTX, and unmapped libdevice functions.
- Gluon, Proton instrumentation, and PTX/TTGIR/LLIR inspection or override.
- Ordinary gather on large tiles can be significantly slower than the NVIDIA PTX backend.

Support depends on the operation, dtype, shape, hardware, and CUDA toolchain version. Automatic backend fallback is disabled by default.

## Install and run

```bash
pip install -e .
ENABLE_TILE=1 python your_program.py
```

Use an NVIDIA driver compatible with the selected CUDA toolchain. Setting `ENABLE_TILE` after importing Triton does not reliably switch an initialized backend.

## Performance

Tune TileIR configurations independently from NVIDIA PTX configurations. `occupancy` and `num_ctas` are useful tuning controls; compatibility parameters do not always have identical semantics across backends. See [Performance Tuning Tips](third_party/tileir/PerformanceTuningTips.md).

## ⚡ Helion Hackathon — Performance Tuning Guide

This source branch uses Triton 3.7 with the CUDA Tile IR 13.4 toolchain; see the [backend build guide](third_party/tileir/README.md).

**The default backend is OSS PTX. The following Triton 3.6 wheel is a previous release. Using [Helion](https://github.com/pytorch/helion) with the TileIR backend([whl](https://github.com/triton-lang/Triton-to-tile-IR/releases/download/v3.6.0-rc1/nvtriton-3.6.0-cp313-cp313-linux_x86_64.whl))?** Check out the **[Helion TileIR Backend Performance Tuning Guide](HelionPerformanceTuningGuide.md)** for config recipes, autotuning strategies, and porting tips.

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

