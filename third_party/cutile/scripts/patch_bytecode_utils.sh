#!/usr/bin/env bash
set -euo pipefail

# Treat the argument as the extracted repo root; construct the target file path.
ARG_PATH="${1:-}"
if [[ -z "${ARG_PATH}" ]]; then
  ARG_PATH="${CUDA_TILE_SOURCE_DIR:-}"
fi

if [[ -z "${ARG_PATH}" ]]; then
  echo "Base directory not provided and CUDA_TILE_SOURCE_DIR unset" >&2
  exit 1
fi

if [[ "${ARG_PATH}" == *.cpp ]]; then
  TARGET_FILE="${ARG_PATH}"
else
  TARGET_FILE="${ARG_PATH}/tools/cuda-tile-tblgen/BytecodeGenUtilities.cpp"
fi

if [[ ! -f "${TARGET_FILE}" ]]; then
  echo "Target file not found: ${TARGET_FILE}" >&2
  exit 1
fi

if [[ ! -f "${TARGET_FILE}.bak" ]]; then
  cp "${TARGET_FILE}" "${TARGET_FILE}.bak"
fi

tmpfile="${TARGET_FILE}.tmp"
trap 'rm -f "${tmpfile}"' EXIT

sed -e 's/getArgToOperandOrAttribute/getArgToOperandAttrOrProp/g' \
    -e 's/OperandOrAttribute/OperandAttrOrProp/g' \
    "${TARGET_FILE}" > "${tmpfile}" && mv "${tmpfile}" "${TARGET_FILE}"

exit 0
