# Triton-TileIR Backend User Guide

This branch uses Triton 3.7, public CUDA Tile IR v13.4.0 sources, and matching 13.4.59 tileiras, ptxas, libnvvm and libdevice components. The source pin and runtime package versions are updated together.

## Build and run

```bash
pip install .
ENABLE_TILE=1 python your_program.py
```

Set `ENABLE_TILE=1` before importing Triton. Without it, the default backend is NVIDIA PTX. A compatible NVIDIA driver is required.

## Capabilities and limitations

The [repository README](../../README.md#supported-operations-and-features) is the single release support list. It retains the original guide, updates supported features, and lists the remaining operation and dtype limitations.

For tuning controls, see [Performance Tuning Tips](PerformanceTuningTips.md). Backend-specific capability expectations are maintained in `python/test/conftest.py`; numerical failures and unexpected compiler errors remain failures.
