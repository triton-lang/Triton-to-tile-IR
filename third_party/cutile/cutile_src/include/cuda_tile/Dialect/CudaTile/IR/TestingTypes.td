#ifndef CUDATILE_DIALECT_CUDATILE_IR_TESTINGTYPES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_TESTINGTYPES_TD

include "mlir/IR/OpBase.td"
include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// Testing Types - Only available when TILE_IR_INCLUDE_TESTS is defined
//===----------------------------------------------------------------------===//

// Testing type definition helper.
class CudaTileTestingTypeDef<string name, string mnemonic, string specName, 
                              string version = "250.0", list<Trait> traits = []>
  : CudaTileTypeDef<name, "testing$" # mnemonic, specName, version, traits>;

//===----------------------------------------------------------------------===//
// BytecodeTest_EvolvedType - Type that evolved from 250.0 to 250.1
//===----------------------------------------------------------------------===//

def CudaTile_BytecodeTest_EvolvedType : CudaTileTestingTypeDef<
    "BytecodeTestEvolved", "bytecode_test_evolved", "bytecode_test_evolved"> {
  let summary = "Testing type for parameter evolution (250.0 -> 250.1)";
  let description = [{
    Type that evolved across versions:
    - Version 250.0: Only has `baseType` 
    - Version 250.1: Added `newType` (DefaultValuedParameter with explicit default i32)
    - Version 250.1: Added `optionalAttr` (OptionalParameter for attribute)
    - Version 250.1: Added `optionalType` (OptionalParameter for Type)
  }];

  let parameters = (ins
    CudaTileConstrainedTypeParam<CudaTile_NumberType, "250.0">:$baseType,
    CudaTileTypeParam<DefaultValuedParameter<"::mlir::Type", "::mlir::IntegerType::get($_ctxt, 32)", "new type defaults to i32">, "250.1">:$newType,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PaddingValueAttr", "optional attribute">, "250.1">:$optionalAttr,
    CudaTileTypeParam<OptionalParameter<"::mlir::Type", "optional type parameter">, "250.1">:$optionalType
  );

  let mnemonic = "testing$bytecode_test_evolved";
  let assemblyFormat = "`<` $baseType (`,` $newType^)? (`,` $optionalAttr^)? (`,` $optionalType^)? `>`";
}

#endif // CUDATILE_DIALECT_CUDATILE_IR_TESTINGTYPES_TD

