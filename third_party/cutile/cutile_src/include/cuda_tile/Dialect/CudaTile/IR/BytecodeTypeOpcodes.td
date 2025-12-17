//===----------------------------------------------------------------------===//
// Bytecode Type Tag Assignments for CudaTile Types
// This file defines the explicit type tag assignments to ensure backward
// compatibility across versions.
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_TYPE_OPCODES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_TYPE_OPCODES_TD

include "cuda_tile/Dialect/CudaTile/IR/Types.td"

//===----------------------------------------------------------------------===//
// Type Tag Assignment Class.
//===----------------------------------------------------------------------===//

/// Base class for type tag assignments.
class BytecodeTypeTag<string typeName, int tag> {
  string cppTypeName = typeName;
  int typeTagValue = tag;
}

/// Integer type tag.
class IntegerTypeTag<string name, int tag, int width>
    : BytecodeTypeTag<name, tag> {
  int integerBitWidth = width;
}

/// Float type tag.
class FloatTypeTag<string name, int tag, string floatType = "">
    : BytecodeTypeTag<name, tag> {
  string floatMlirTypeName = floatType;
}

/// CudaTile type tag.
class CudaTileTypeTag<string name, int tag>
  : BytecodeTypeTag<name, tag> {
}

//===----------------------------------------------------------------------===//
// Explicit Type Tag Assignments - FROZEN for backward compatibility.
//===----------------------------------------------------------------------===//

// 13.1 types.
def : IntegerTypeTag<"Int1", 0, 1>;
def : IntegerTypeTag<"Int8", 1, 8>;
def : IntegerTypeTag<"Int16", 2, 16>;
def : IntegerTypeTag<"Int32", 3, 32>;
def : IntegerTypeTag<"Int64", 4, 64>;
def : FloatTypeTag<"Float16", 5, "Float16Type">;
def : FloatTypeTag<"BFloat16", 6, "BFloat16Type">;
def : FloatTypeTag<"Float32", 7, "Float32Type">;
def : FloatTypeTag<"TFloat32", 8, "FloatTF32Type">;
def : FloatTypeTag<"Float64", 9, "Float64Type">;
def : FloatTypeTag<"Float8E4M3FN", 10, "Float8E4M3FNType">;
def : FloatTypeTag<"Float8E5M2", 11, "Float8E5M2Type">;
def : CudaTileTypeTag<"PointerType", 12>;
def : CudaTileTypeTag<"TileType", 13>;
def : CudaTileTypeTag<"TensorViewType", 14>;
def : CudaTileTypeTag<"PartitionViewType", 15>;
def : CudaTileTypeTag<"FunctionType", 16>;
def : CudaTileTypeTag<"TokenType", 17>;

// 13.2 types.
def : FloatTypeTag<"Float8E8M0FNU", 18, "Float8E8M0FNUType">;

#ifdef TILE_IR_INCLUDE_TESTS
// Testing type for parameter versioning evolution (version from TestingTypes.td).
def : CudaTileTypeTag<"BytecodeTestEvolvedType", 200>;
#endif // TILE_IR_INCLUDE_TESTS


#endif // CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_TYPE_OPCODES_TD
