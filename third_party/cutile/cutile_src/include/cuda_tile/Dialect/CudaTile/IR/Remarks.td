//===- Remarks.td - CUDA Tile Remark Definitions ------------*- tablegen -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_REMARKS_TD
#define CUDATILE_DIALECT_REMARKS_TD

include "cuda_tile/Dialect/CudaTile/IR/RemarkUtils.td"

//===--------------------------------------------------------------------===//
// Remark Categories
//===--------------------------------------------------------------------===//

def CategoryTensorCore : TileIRCategory<"TensorCore", 1, "Tensor-core">;
def CategoryMemory     : TileIRCategory<"Memory",     2, "Memory">;

//===--------------------------------------------------------------------===//
// Remarks
//===--------------------------------------------------------------------===//

// Tensor Core Remarks
def RemarkTensorCoreMMAOptimized
  : TileIRRemark<"RemarkTensorCoreMMAOptimized", CategoryTensorCore, 1, "MMA operation successfully optimized to use Tensor Cores">;

// Memory Remarks
def RemarkMemoryLoadOptimized
  : TileIRRemark<"RemarkMemoryLoadOptimized", CategoryMemory, 1, "Load operation successfully optimized to use TMA">;
def RemarkMemoryStoreOptimized
  : TileIRRemark<"RemarkMemoryStoreOptimized", CategoryMemory, 2, "Store operation successfully optimized to use TMA">;

//===--------------------------------------------------------------------===//
// Attribute Definitions
//===--------------------------------------------------------------------===//
def CategoryID : I32EnumAttr<"CategoryID",
                             "TileIR remark categories",
                             [CategoryTensorCore, CategoryMemory]> {
  let cppNamespace        = "::mlir::cuda_tile::remark";
  let genSpecializedAttr  = 0;
}
def RemarkID
    : I32EnumAttr<"RemarkID", "All TileIR remark IDs",
                  [RemarkTensorCoreMMAOptimized, 
                   RemarkMemoryLoadOptimized, 
                   RemarkMemoryStoreOptimized
                   /*, ...*/]> {
  let cppNamespace = "::mlir::cuda_tile::remark";
  let genSpecializedAttr  = 0;
}

#endif
