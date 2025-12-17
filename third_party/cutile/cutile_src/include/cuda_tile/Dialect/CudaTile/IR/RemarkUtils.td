//===- RemarkUtils.td - CUDA Tile Remark Utility Definitions ------------*- tablegen -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Properties.td"

/// Remark Category Definitions
class TileIRCategoryCase<string Name, int Val, string Doc> : I32EnumAttrCase<Name, Val, Doc>{}

/// Remark Category Definition
class TileIRCategory<string Name, int Id, string Doc> : I32EnumAttrCase<Name, Id, Doc>;

/// Remark Definition
class TileIRRemark<string Name, TileIRCategory Cat, int Index, string Doc>
  : I32EnumAttrCase<
        Name,
        !or(!shl(Cat.value, 10), Index),
        Doc>;
