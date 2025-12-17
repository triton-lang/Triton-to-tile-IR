//===----------------------------------------------------------------------===//
// This file defines pure code generation classes for type bytecode.
// These classes operate on pre-analyzed BytecodeTypeStructure data and
// do not directly access TableGen records.
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_TBLGEN_BYTECODE_TYPE_CODEGEN_H_
#define CUDA_TILE_TOOLS_TBLGEN_BYTECODE_TYPE_CODEGEN_H_

#include "llvm/Support/raw_ostream.h"

#include "BytecodeTypeAnalysis.h"

namespace mlir {
namespace tblgen {

//===----------------------------------------------------------------------===//
// C++ Code Generation Functions.
//===----------------------------------------------------------------------===//

/// Generate type tag enum.
void generateTypeTagEnum(const BytecodeTypeStructure &structure,
                         llvm::raw_ostream &os);

/// Generate type serialization functions.
void generateTypeSerializers(const BytecodeTypeStructure &structure,
                             llvm::raw_ostream &os);

/// Generate type deserialization functions.
void generateTypeDeserializers(const BytecodeTypeStructure &structure,
                               llvm::raw_ostream &os);

/// Generate serialization dispatch logic.
void generateSerializerDispatch(const BytecodeTypeStructure &structure,
                                llvm::raw_ostream &os);

/// Generate deserialization dispatch logic.
void generateDeserializerDispatch(const BytecodeTypeStructure &structure,
                                  llvm::raw_ostream &os);

/// Generate dependent type registration logic for TypeManager.
void generateDependentTypeRegistration(const BytecodeTypeStructure &structure,
                                       llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // CUDA_TILE_TOOLS_TBLGEN_BYTECODE_TYPE_CODEGEN_H_
