# nvtriton — Triton with TileIR Backend

Install nvtriton alongside your existing Triton. OSS Triton is never modified.

## Install

```bash
bash install_nvtriton.sh                  # default: ~/.local/triton_tileir
bash install_nvtriton.sh /my/custom/path  # custom location
```

## Activate / Deactivate

```bash
source <install_dir>/activate.sh    # enable TileIR backend
source <install_dir>/deactivate.sh  # revert to OSS Triton
```

## Uninstall

```bash
bash uninstall_nvtriton.sh                  # default path
bash uninstall_nvtriton.sh /my/custom/path  # custom path
```

> Deactivate before uninstalling — the script will remind you if you forget.
