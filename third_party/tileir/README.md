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

- Native scaled MMA supports matching FP4/FP8 operand types with both scales; mixed types and omitted scales are not supported.
- Descriptor gather/scatter is supported; ordinary `tl.gather` and histogram have no lowering in this backend.
- Conditional branches cannot return descriptor views in the public 13.4 dialect.
- Block pointers, descriptor atomic reduction and general inline assembly remain unsupported. The CUDA GDC helpers have dedicated native lowering.
- Backend-specific capability expectations are recorded in `python/test/conftest.py`. Numerical failures and unexpected compiler errors remain failures; missing PTX/TTGIR inspection does not establish numerical coverage.
