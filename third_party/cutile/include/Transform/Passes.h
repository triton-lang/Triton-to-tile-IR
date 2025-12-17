#ifndef TRITON_CUTILE_TRANSFORMS_PASSES_H_
#define TRITON_CUTILE_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <optional>

namespace mlir {
namespace triton {
std::unique_ptr<Pass>
createRewriteTensorPointerToMemrefPass(int computeCapability,
                                       std::optional<int> numStages);
std::unique_ptr<Pass> createRewriteAssumeWithCudaTilePass();
std::unique_ptr<Pass> createLiftTTCFToSCFPass();
std::unique_ptr<Pass> createAutoGenMemoryTokenPass();
std::unique_ptr<Pass> createAutoGenMemoryTokenPass(bool enable_autogen_alias_mem_token);

#define GEN_PASS_REGISTRATION
#include "Transform/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif // TRITON_CUTILE_TRANSFORMS_PASSES_H_
