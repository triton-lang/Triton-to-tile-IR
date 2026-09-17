#!/usr/bin/env bash
# Uninstall triton_tileir completely.
#
# Usage:
#   bash uninstall_triton_tileir.sh              # removes ~/.local/triton_tileir
#   bash uninstall_triton_tileir.sh /my/path     # removes /my/path
set -euo pipefail

INSTALL_DIR="${1:-${HOME}/.local/triton_tileir}"

# Check if triton_tileir is still active in current shell
if [ -n "${ENABLE_TILE:-}" ] || echo "${PYTHONPATH:-}" | grep -q "${INSTALL_DIR}"; then
    echo "Error: triton_tileir is still active. Please deactivate first:"
    echo "  source ${INSTALL_DIR}/deactivate.sh"
    exit 1
fi

if [ -d "${INSTALL_DIR}" ]; then
    [[ -f "${INSTALL_DIR}/.triton-tileir-install" ]] || { echo "Refusing to remove a directory not created by this installer." >&2; exit 2; }
    rm -rf -- "${INSTALL_DIR}"
    echo "==> Removed ${INSTALL_DIR}"
else
    echo "==> ${INSTALL_DIR} not found (already clean)"
fi

echo "Done!"
