# Installing nvtriton alongside upstream Triton

This guide explains how to install nvtriton (Triton with TileIR backend) side-by-side with
upstream OpenAI Triton (oait) and switch between them using environment variables.

## Prerequisites

- Python 3.12 or 3.13 virtual environment with PyTorch and triton already installed
- nvtriton wheel from the [release page](https://github.com/triton-lang/Triton-to-tile-IR/releases/tag/v3.6.0-rc1)
  - `nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl` for Python 3.12
  - `nvtriton-3.6.0-cp313-cp313-linux_x86_64.whl` for Python 3.13

## Installation

Install nvtriton into an isolated directory so it does not overwrite the existing triton:

```bash
NVTRITON_DIR=$VIRTUAL_ENV/opt/nvtriton   # or ~/.local/nvtriton, /opt/nvtriton, etc.

mkdir -p $NVTRITON_DIR
pip install --no-cache-dir --no-deps --target $NVTRITON_DIR ./nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl
```

`--no-deps` is required — nvtriton shares the same dependencies as oait triton, so they
do not need to be installed again.

## Usage

### Default: upstream triton (oait)

No changes needed. Python uses the triton in site-packages:

```bash
python my_script.py
```

### Switch to nvtriton (TileIR backend)

Prepend `PYTHONPATH` and set `ENABLE_TILE=1`:

```bash
PYTHONPATH=$NVTRITON_DIR ENABLE_TILE=1 python my_script.py
```

Or export for the current shell session:

```bash
export PYTHONPATH=$NVTRITON_DIR
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

# Confirm nvtriton activates via PYTHONPATH
PYTHONPATH=$NVTRITON_DIR ENABLE_TILE=1 \
  python -c "import triton; print(triton.__file__)"
# → .../opt/nvtriton/triton/__init__.py

# Confirm TileIRDriver is active
PYTHONPATH=$NVTRITON_DIR ENABLE_TILE=1 \
  python -c "from triton.runtime.driver import driver; print(type(driver.active).__name__)"
# → TileIRDriver
```

## How it works

`PYTHONPATH` entries are searched before `site-packages`. When set to `$NVTRITON_DIR`,
Python finds `$NVTRITON_DIR/triton/` first, which shadows the oait `triton/` in
site-packages. When unset, Python falls back to the default oait triton. The two
installations are fully isolated — neither modifies the other.

## Docker usage

In Dockerfiles, the same pattern applies:

```dockerfile
# Install oait triton (comes with PyTorch or install explicitly)
RUN pip install triton==3.6.0

# Install nvtriton to /opt/nvtriton
RUN curl -L -o /tmp/nvtriton.whl <wheel-url> \
    && pip install --no-cache-dir --no-deps --target /opt/nvtriton /tmp/nvtriton.whl \
    && rm /tmp/nvtriton.whl
```

Then at runtime:

```bash
# Use oait (default)
docker run myimage python script.py

# Use nvtriton
docker run -e PYTHONPATH=/opt/nvtriton -e ENABLE_TILE=1 myimage python script.py
```

## Notes

- The nvtriton wheel embeds `tileiras` and `ptxas` binaries in
  `triton/backends/tileir/scripts/cuda_dep_x86/`. No separate CUDA toolkit is needed
  for the TileIR backend to function.
- Always use `--no-deps` when installing to `--target`. Otherwise pip copies redundant
  dependencies into the target directory.
- This approach works in conda environments as well — `PYTHONPATH` takes precedence
  regardless of the package manager.
