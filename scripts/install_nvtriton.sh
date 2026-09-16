#!/usr/bin/env bash
# Install nvtriton (Triton with TileIR backend) alongside OSS triton.
# OSS triton is untouched.
#
# Usage:
#   bash install_nvtriton.sh              # installs to ~/.local/triton_tileir
#   bash install_nvtriton.sh /my/path     # installs to /my/path
#
# After install, activate with:
#   source <install_dir>/activate.sh
set -euo pipefail

RELEASE_URL="https://github.com/triton-lang/Triton-to-tile-IR/releases/download/v3.6.0-rc1"
INSTALL_DIR="${1:-${HOME}/.local/triton_tileir}"

PY_TAG=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
WHEEL="nvtriton-3.6.0-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"

echo "==> Detected Python ${PY_TAG}"
echo "==> Install directory: ${INSTALL_DIR}"

echo "==> Downloading ${WHEEL}..."
curl -fSL -o "/tmp/${WHEEL}" "${RELEASE_URL}/${WHEEL}"

echo "==> Installing to ${INSTALL_DIR} (OSS triton is untouched)..."
mkdir -p "${INSTALL_DIR}"
python3 -m pip install --no-cache-dir --no-deps --target "${INSTALL_DIR}" "/tmp/${WHEEL}"
rm -f "/tmp/${WHEEL}"

# Generate activate.sh
cat > "${INSTALL_DIR}/activate.sh" <<EOF
# Source this file to enable nvtriton TileIR backend.
#   source ${INSTALL_DIR}/activate.sh
export PYTHONPATH="${INSTALL_DIR}\${PYTHONPATH:+:\$PYTHONPATH}"
export ENABLE_TILE=1
export HELION_BACKEND=tileir
echo "nvtriton activated."
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
unset HELION_BACKEND
echo "nvtriton deactivated. OSS triton is now active."
EOF

echo ""
echo "Done! To activate:   source ${INSTALL_DIR}/activate.sh"
echo "      To deactivate: source ${INSTALL_DIR}/deactivate.sh"
echo "      To uninstall:  bash uninstall_nvtriton.sh ${INSTALL_DIR}"
