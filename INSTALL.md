# Installing triton_tileir alongside upstream Triton

This guide explains how to install triton_tileir (Triton with TileIR backend) side-by-side with
upstream OpenAI Triton (oait) and switch between them using environment variables.

## Prerequisites

- Python 3.12 virtual environment with PyTorch and triton already installed
- triton_tileir wheel from the [release page](https://github.com/triton-lang/Triton-to-tile-IR/releases)
  - `triton_tileir-3.6.0+tileir13.4.0-cp312-cp312-linux_x86_64.whl` for Python 3.12

## Installation

Install triton_tileir into an isolated directory so it does not overwrite the existing triton:

```bash
TRITON_TILEIR_DIR=$VIRTUAL_ENV/opt/triton_tileir   # or ~/.local/triton_tileir, /opt/triton_tileir, etc.

mkdir -p $TRITON_TILEIR_DIR
pip install --no-cache-dir --no-deps --target $TRITON_TILEIR_DIR ./triton_tileir-3.6.0+tileir13.4.0-cp312-cp312-linux_x86_64.whl
```

`--no-deps` is required — triton_tileir shares the same dependencies as oait triton, so they
do not need to be installed again.

## Usage

### Default: upstream triton (oait)

No changes needed. Python uses the triton in site-packages:

```bash
python my_script.py
```

### Switch to triton_tileir (TileIR backend)

Prepend `PYTHONPATH` and set `ENABLE_TILE=1`:

```bash
PYTHONPATH=$TRITON_TILEIR_DIR ENABLE_TILE=1 python my_script.py
```

Or export for the current shell session:

```bash
export PYTHONPATH=$TRITON_TILEIR_DIR
export ENABLE_TILE=1
python my_script.py

# revert when done
unset PYTHONPATH ENABLE_TILE
```

### Switch back to oait

Simply unset the variables (or start a new shell):

```bash
unset PYTHONPATH ENABLE_TILE
python my_script.py   # back to oait
```

## Verification

```bash
# Confirm oait is the default
python -c "import triton; print(triton.__file__)"
# → .../site-packages/triton/__init__.py

# Confirm triton_tileir activates via PYTHONPATH
PYTHONPATH=$TRITON_TILEIR_DIR ENABLE_TILE=1 \
  python -c "import triton; print(triton.__file__)"
# → .../opt/triton_tileir/triton/__init__.py

# Confirm TileIRDriver is active
PYTHONPATH=$TRITON_TILEIR_DIR ENABLE_TILE=1 \
  python -c "from triton.runtime.driver import driver; print(type(driver.active).__name__)"
# → TileIRDriver
```

## How it works

`PYTHONPATH` entries are searched before `site-packages`. When set to `$TRITON_TILEIR_DIR`,
Python finds `$TRITON_TILEIR_DIR/triton/` first, which shadows the oait `triton/` in
site-packages. When unset, Python falls back to the default oait triton. The two
installations are fully isolated — neither modifies the other.

## Docker usage

In Dockerfiles, the same pattern applies:

```dockerfile
# Install oait triton (comes with PyTorch or install explicitly)
RUN pip install triton==3.6.0

# Install triton_tileir to /opt/triton_tileir
COPY triton_tileir-3.6.0+tileir13.4.0-cp312-cp312-linux_x86_64.whl /tmp/
RUN pip install --no-cache-dir --no-deps --target /opt/triton_tileir /tmp/triton_tileir-3.6.0+tileir13.4.0-cp312-cp312-linux_x86_64.whl \
    && rm /tmp/triton_tileir-3.6.0+tileir13.4.0-cp312-cp312-linux_x86_64.whl
```

Then at runtime:

```bash
# Use oait (default)
docker run myimage python script.py

# Use triton_tileir
docker run -e PYTHONPATH=/opt/triton_tileir -e ENABLE_TILE=1 myimage python script.py
```

## Notes

- The triton_tileir wheel embeds `tileiras` and `ptxas` binaries in
  `triton/backends/nvidia/tileir_cuda/`. No separate CUDA toolkit is needed
  for the TileIR backend to function.
- Always use `--no-deps` when installing to `--target`. Otherwise pip copies redundant
  dependencies into the target directory.
- This approach works in conda environments as well — `PYTHONPATH` takes precedence
  regardless of the package manager.
