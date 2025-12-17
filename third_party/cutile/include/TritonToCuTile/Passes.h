#ifndef TRITON_TO_CUTILE_CONVERSION_PASSES_H
#define TRITON_TO_CUTILE_CONVERSION_PASSES_H

#include "TritonToCuTile/TritonToCuTilePass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "TritonToCuTile/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CUTILE_CONVERSION_PASSES_H
