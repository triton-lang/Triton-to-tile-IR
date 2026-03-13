# nvtriton — Triton with TileIR Backend

Install nvtriton alongside your existing Triton. OSS Triton is never modified.

## Quick Start

```bash
bash install_nvtriton.sh                          # install
source ~/.local/triton_tileir/activate.sh         # activate
source ~/.local/triton_tileir/deactivate.sh       # deactivate
bash uninstall_nvtriton.sh                        # uninstall
```

## Custom Install Path

```bash
bash install_nvtriton.sh /my/custom/path
source /my/custom/path/activate.sh
source /my/custom/path/deactivate.sh
bash uninstall_nvtriton.sh /my/custom/path
```

> Deactivate before uninstalling — the script will remind you if you forget.
