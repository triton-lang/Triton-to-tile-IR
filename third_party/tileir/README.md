# Triton-TileIR Backend User Guide

## Build Instructions

To build and install the Triton-TileIR backend, simply run:

```bash
pip install .
```

This branch uses a Triton 3.8 development snapshot, public CUDA Tile IR v13.4.0 sources, and bundled CUDA 13.4.59 compiler and runtime components.

## Running

Before using the backend, ensure your NVIDIA driver is compatible with CUDA 13.4 and set the following environment variable before importing Triton:

```bash
export ENABLE_TILE=1
```

## Known Limitations

- Some tests that are not supported by CudaTile are not yet automatically skipped; as a result, you may see failures in certain unit tests.

For the current feature support list, see the [repository README](../../README.md#supported-operations-and-features). For tuning guidance, see [Performance Tuning Tips](PerformanceTuningTips.md).
