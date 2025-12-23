#ifndef TRITON_TO_TILEIR_CONVERSION_PASSES_H
#define TRITON_TO_TILEIR_CONVERSION_PASSES_H

#include "TritonToTileIR/TritonToTileIRPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "TritonToTileIR/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_TO_TILEIR_CONVERSION_PASSES_H
