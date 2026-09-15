# Triton-TileIR Backend User Guide

## Build Instructions

To build and install the Triton-TileIR backend, simply run:

```bash
pip install .
```

## Running

This source branch uses the public CUDA Tile IR 13.4 sources and bundles the matching 13.4.59 tileiras, ptxas, libnvvm and libdevice components at build time. A compatible NVIDIA driver is required. Enable the backend before importing Triton:

```bash
export ENABLE_TILE=1
```

## Known Limitations

- Native scaled MMA supports matching FP4/FP8 operand types with both scales, and matching FP8 with exactly one scale. Mixed operand types, single-scale FP4 and two omitted scales remain unsupported.
- Descriptor gather/scatter and ordinary `tl.gather` are supported. Histogram lowering remains unavailable.
- Conditional branches cannot return descriptor views in the public 13.4 dialect.
- Descriptor atomic reduction supports integer operations and floating-point add; floating-point min/max remain unsupported.
- Block pointers and general inline assembly remain unsupported. CUDA GDC helpers and a small set of explicitly matched numerical assembly forms have native lowering.
- Source line information is available, but some optimized loop and inlined helper lines are not preserved.
- Backend-specific capability expectations are recorded in `python/test/conftest.py`. Numerical failures and unexpected compiler errors remain failures; missing PTX/TTGIR inspection does not establish numerical coverage.
