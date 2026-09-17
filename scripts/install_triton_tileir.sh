#!/usr/bin/env bash
# Install triton_tileir (Triton with TileIR backend) alongside OSS triton.
# OSS triton is untouched.
#
# Usage:
#   bash install_triton_tileir.sh WHEEL        # installs to ~/.local/triton_tileir
#   bash install_triton_tileir.sh WHEEL /my/path     # installs to /my/path
#
# After install, activate with:
#   source <install_dir>/activate.sh
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || { echo "usage: $0 LOCAL_WHEEL [INSTALL_DIR]" >&2; exit 2; }
WHEEL=$(realpath "$1")
[[ -f "$WHEEL" && "${WHEEL##*/}" == triton_tileir-*.whl ]] || { echo "Select a downloaded Triton TileIR wheel." >&2; exit 2; }
INSTALL_DIR="${2:-${HOME}/.local/triton_tileir}"
[[ "$INSTALL_DIR" != *'"'* && "$INSTALL_DIR" != *'`'* && "$INSTALL_DIR" != *'$'* && "$INSTALL_DIR" != *$'\n'* ]] || { echo "Unsupported install path characters." >&2; exit 2; }
INSTALL_DIR=$(realpath -m "$INSTALL_DIR")
[[ ! -e "$INSTALL_DIR" ]] || { echo "Choose a new install directory." >&2; exit 2; }
mkdir -p "$INSTALL_DIR"
python3 -m pip install --no-cache-dir --no-deps --target "$INSTALL_DIR" "$WHEEL"
printf '%s\n' "Triton TileIR isolated installation" > "$INSTALL_DIR/.triton-tileir-install"

# Generate activate.sh
cat > "${INSTALL_DIR}/activate.sh" <<EOF
# Source this file to enable triton_tileir TileIR backend.
#   source ${INSTALL_DIR}/activate.sh
export PYTHONPATH="${INSTALL_DIR}\${PYTHONPATH:+:\$PYTHONPATH}"
export ENABLE_TILE=1
echo "triton_tileir activated."
EOF

# Generate deactivate.sh
cat > "${INSTALL_DIR}/deactivate.sh" <<EOF
# Source this file to revert to OSS triton.
#   source ${INSTALL_DIR}/deactivate.sh
if [ -n "\${PYTHONPATH:-}" ]; then
    PYTHONPATH=\$(echo "\${PYTHONPATH}" | tr ':' '\n' | grep -v "^${INSTALL_DIR}\\\$" | paste -sd ':' || true)
    [ -z "\${PYTHONPATH}" ] && unset PYTHONPATH || export PYTHONPATH
fi
unset ENABLE_TILE
echo "triton_tileir deactivated. OSS triton is now active."
EOF

echo ""
echo "Done! To activate:   source ${INSTALL_DIR}/activate.sh"
echo "      To deactivate: source ${INSTALL_DIR}/deactivate.sh"
echo "      To uninstall:  bash uninstall_triton_tileir.sh ${INSTALL_DIR}"
