#!/usr/bin/env bash
# Uninstall nvtriton completely.
#
# Usage:
#   bash uninstall_nvtriton.sh              # removes ~/.local/triton_tileir
#   bash uninstall_nvtriton.sh /my/path     # removes /my/path
set -euo pipefail

INSTALL_DIR="${1:-${HOME}/.local/triton_tileir}"

# Check if nvtriton is still active in current shell
if [ -n "${ENABLE_TILE:-}" ] || [ -n "${HELION_BACKEND:-}" ] || echo "${PYTHONPATH:-}" | grep -q "${INSTALL_DIR}"; then
    echo "Error: nvtriton is still active. Please deactivate first:"
    echo "  source ${INSTALL_DIR}/deactivate.sh"
    exit 1
fi

if [ -d "${INSTALL_DIR}" ]; then
    rm -rf "${INSTALL_DIR}"
    echo "==> Removed ${INSTALL_DIR}"
else
    echo "==> ${INSTALL_DIR} not found (already clean)"
fi

echo "Done!"
