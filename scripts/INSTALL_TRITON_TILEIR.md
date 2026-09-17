# triton_tileir — Triton with TileIR Backend

Install triton_tileir alongside your existing Triton. OSS Triton is never modified.

## Quick Start

```bash
bash install_triton_tileir.sh /path/to/downloaded-wheel.whl # install
source ~/.local/triton_tileir/activate.sh         # activate
source ~/.local/triton_tileir/deactivate.sh       # deactivate
bash uninstall_triton_tileir.sh                        # uninstall
```

## Custom Install Path

```bash
bash install_triton_tileir.sh /path/to/downloaded-wheel.whl /my/custom/path
source /my/custom/path/activate.sh
source /my/custom/path/deactivate.sh
bash uninstall_triton_tileir.sh /my/custom/path
```

> Deactivate before uninstalling — the script will remind you if you forget.

Download the wheel matching your Python and platform from the [release page](https://github.com/triton-lang/Triton-to-tile-IR/releases). The installer takes a local wheel and never selects an older release automatically.
