#!/usr/bin/env bash
set -euo pipefail

patch_in_place() {
  local file="$1"; shift
  if [[ ! -f "${file}" ]]; then
    echo "[patch] Target file not found: ${file}" >&2
    exit 1
  fi

  if [[ ! -f "${file}.bak" ]]; then
    cp "${file}" "${file}.bak"
  fi

  local tmpfile="${file}.tmp"
  rm -f "${tmpfile}"

  # Keep each sed argument intact (some expressions include spaces).
  sed "$@" "${file}" > "${tmpfile}" && mv "${tmpfile}" "${file}"
}

# Treat the argument as the extracted cuda_tile repo root (preferred), or fall back
# to CUDA_TILE_SOURCE_DIR (used by CMake).
ARG_PATH="${1:-${CUDA_TILE_SOURCE_DIR:-}}"
if [[ -z "${ARG_PATH}" ]]; then
  echo "[patch] Base directory not provided and CUDA_TILE_SOURCE_DIR unset" >&2
  exit 1
fi

# Allow passing either repo root or a direct file path (legacy behavior).
if [[ "${ARG_PATH}" == *.cpp || "${ARG_PATH}" == *.td ]]; then
  REPO_ROOT="$(cd "$(dirname "${ARG_PATH}")/.." && pwd)"
else
  REPO_ROOT="${ARG_PATH}"
fi

BYTECODE_UTIL_PATH="${REPO_ROOT}/tools/cuda-tile-tblgen/BytecodeGenUtilities.cpp"
OPS_TD_PATH="${REPO_ROOT}/include/cuda_tile/Dialect/CudaTile/IR/Ops.td"
CUDATILE_CPP_PATH="${REPO_ROOT}/lib/Dialect/CudaTile/IR/CudaTile.cpp"

echo "[patch] repo_root=${REPO_ROOT}"

# 1) Patch BytecodeGenUtilities.cpp for LLVM api changes:
# Replace "getArgToOperandOrAttribute" with "getArgToOperandAttrOrProp"
# and "OperandOrAttribute" with "OperandAttrOrProp".
if [[ -f "${BYTECODE_UTIL_PATH}" ]]; then
  echo "[patch] Patching: ${BYTECODE_UTIL_PATH}"
  patch_in_place "${BYTECODE_UTIL_PATH}" \
    -e 's/getArgToOperandOrAttribute/getArgToOperandAttrOrProp/g' \
    -e 's/OperandOrAttribute/OperandAttrOrProp/g'
fi

# 2) Patch Ops.td for LLVM api changes:
# - replace 'CArg<"ValueRange", "std::nullopt">:$initArgs' with 'CArg<"ValueRange", "{}">:$initArgs'
# - replace 'build($_builder, $_state, std::nullopt)' with 'build($_builder, $_state, ::mlir::ValueRange{})'
if [[ -f "${OPS_TD_PATH}" ]]; then
  echo "[patch] Patching: ${OPS_TD_PATH}"
  patch_in_place "${OPS_TD_PATH}" \
    -e 's/CArg<"ValueRange", "std::nullopt">:$initArgs/CArg<"ValueRange", "{}">:$initArgs/g' \
    -e 's/build($_builder, $_state, std::nullopt)/build($_builder, $_state, ::mlir::ValueRange{})/g'
fi

# 3) Patch CudaTile.cpp for LLVM api changes:
# replace 'ValueRange(), /*attributes=*/std::nullopt)' with
# 'ValueRange(), /*attributes=*/llvm::ArrayRef<mlir::NamedAttribute>{})'
if [[ -f "${CUDATILE_CPP_PATH}" ]]; then
  echo "[patch] Patching: ${CUDATILE_CPP_PATH}"
  patch_in_place "${CUDATILE_CPP_PATH}" \
    -e 's|ValueRange(), /\*attributes=\*/std::nullopt)|ValueRange(), /\*attributes=\*/llvm::ArrayRef<mlir::NamedAttribute>{})|g'
fi

# 4) Triton 3.7's LLVM renamed DenseIntOrFPElementsAttr to
# DenseTypedElementsAttr. Keep TileIR 13.3 pinned and bridge the copied build
# source instead of changing the released TileIR sources.
echo "[patch] Global rename: DenseIntOrFPElementsAttr → DenseTypedElementsAttr"
find "${REPO_ROOT}" -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.td" \) \
  -exec sed -i 's/DenseIntOrFPElementsAttr/DenseTypedElementsAttr/g' {} +

# DenseElementsAttr<i1> raw layout differs across the LLVM versions used by
# Triton and the bundled tileiras. Patch only the copied cuda-tile source tree.
python3 "$(dirname "${BASH_SOURCE[0]}")/patch_cuda_tile_i1_bytecode_compat.py" "${REPO_ROOT}"

# 5) Patch BytecodeReader.cpp for LLVM api changes:
# - Triton 3.7's LLVM uses the 2-argument isValidRawBuffer overload.
# - Triton 3.7's LLVM exposes make_scope_exit rather than a directly
#   constructible scope_exit template.
BYTECODE_READER_PATH="${REPO_ROOT}/lib/Bytecode/Reader/BytecodeReader.cpp"
if [[ -f "${BYTECODE_READER_PATH}" ]]; then
  echo "[patch] Patching: ${BYTECODE_READER_PATH}"
  patch_in_place "${BYTECODE_READER_PATH}" \
    -e 's/llvm::scope_exit removeIndex(/auto removeIndex = llvm::make_scope_exit(/g' \
    -e 's/DenseElementsAttr::isValidRawBuffer(tileType, rawData, isSplat)/DenseElementsAttr::isValidRawBuffer(tileType, rawData)/g'
fi

# This option is newer than the LLVM pinned by Triton 3.7.
DIALECT_TD_PATH="${REPO_ROOT}/include/cuda_tile/Dialect/CudaTile/IR/Dialect.td"
if [[ -f "${DIALECT_TD_PATH}" ]] && grep -q 'usePropertiesForAttributes' "${DIALECT_TD_PATH}"; then
  echo "[patch] Removing unsupported usePropertiesForAttributes"
  patch_in_place "${DIALECT_TD_PATH}" -e '/usePropertiesForAttributes/d'
fi

echo "[patch] DONE"
