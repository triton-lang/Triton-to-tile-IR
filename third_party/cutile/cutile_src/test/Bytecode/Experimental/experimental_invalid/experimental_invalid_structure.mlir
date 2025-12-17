// This file contains various failure test cases related to the structure of
// a bytecode file.

//===--------------------------------------------------------------------===//
// Invalid APInt Value
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/apint_validation_issue.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=APINT_VALUE
// APINT_VALUE: value 18446744073709551613 does not fit in 32 bits

//===--------------------------------------------------------------------===//
// Invalid Function Length
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_function_length.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=FUNC_LENGTH
// FUNC_LENGTH: function body length 18446744073709551615 exceeds remaining bytecode data

//===--------------------------------------------------------------------===//
// Infinite Recursion LazyTypeTable
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/infinite_recursion_lazy_type.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=LazyTypeTable
// LazyTypeTable: failed to get parameter type

//===--------------------------------------------------------------------===//
// Invalid DenseElementsAttr Type
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_dense_type.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=DENSE_TYPE
// DENSE_TYPE: invalid block structure: block is expected to have a terminator operation, but it is empty

//===--------------------------------------------------------------------===//
// Invalid number of regions
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_num_regions.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=NUM_REGIONS
// NUM_REGIONS: varint value exceeds maximum supported capacity.
// NUM_REGIONS: failed to read number of regions to parse.

//===--------------------------------------------------------------------===//
// Invalid size of type section
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_type_section_size.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=TYPE_SECTION_SIZE
// TYPE_SECTION_SIZE: number of types (9223372036854775808) exceeds the maximum of 38 that can fit in the remaining payload of 153 bytes.
// TYPE_SECTION_SIZE: failed to parse type section

//===--------------------------------------------------------------------===//
// Invalid size of debug section
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_debug_section_size.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=DEBUG_SECTION_SIZE
// DEBUG_SECTION_SIZE: number of debug info attributes (9223372036854775755) exceeds the maximum of 74 that can fit in the remaining payload of 297 bytes.
// DEBUG_SECTION_SIZE: failed to parse debug section

//===--------------------------------------------------------------------===//
// Invalid size of global section
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_global_section_size.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=GLOBAL_SECTION_SIZE
// GLOBAL_SECTION_SIZE: number of globals (9223372036207994367) exceeds the maximum of 4 that can fit in the remaining payload of 18 bytes.

//===--------------------------------------------------------------------===//
// Invalid section alignment
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_section_alignment.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=SECTION_ALIGNMENT
// SECTION_ALIGNMENT: section 3 must have alignment that is a multiple of 8 bytes, but has alignment of 2

//===--------------------------------------------------------------------===//
// Invalid optimization hints attr
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/bad_optimization_hints_attr.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=OPT_HINT_ATTR
// OPT_HINT_ATTR: unknown param kernel_test_4 for sm_100
// OPT_HINT_ATTR: failed to parse OptimizationHintsAttr

//===--------------------------------------------------------------------===//
// Invalid dense type for global
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_dense_type_global.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=INVALID_DENSE_TYPE_GLOBAL
// INVALID_DENSE_TYPE_GLOBAL: provided type is null
// INVALID_DENSE_TYPE_GLOBAL: failed to create global from bytecode

//===--------------------------------------------------------------------===//
// Corrupted debug info indices (regression test for segfault fix)
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/corrupted_debug_info_indices.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=CORRUPTED_DEBUG_INFO
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/corrupted_debug_info_indices_2.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=CORRUPTED_DEBUG_INFO
// CORRUPTED_DEBUG_INFO: failed to read function location
// CORRUPTED_DEBUG_INFO: failed to create function from bytecode

//===--------------------------------------------------------------------===//
// Invalid array size for ArrayAttr
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_array_size.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=ARRAY_SIZE
// ARRAY_SIZE: varint value exceeds maximum supported capacity. (expected value less than 4294967294, got [[VALUE:[0-9]+]]).
// ARRAY_SIZE: failed to read size for ArrayAttr.
// ARRAY_SIZE: failed to parse attribute 'identities'
// ARRAY_SIZE: failed to parse function body for function 'kernel'
// ARRAY_SIZE: failed to create function from bytecode

//===--------------------------------------------------------------------===//
// ForOp missing valid induction variable
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_for_no_induction_var.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=INVALID_FOR_NO_INDUCTION_VARIABLE
// INVALID_FOR_NO_INDUCTION_VARIABLE: 'cuda_tile.for' op expected at least one block argument for induction variable

//===--------------------------------------------------------------------===//
// PartitionView type invalid tile (too large)
//===--------------------------------------------------------------------===//
// RUN: not cuda-tile-translate -cudatilebc-to-mlir %S/invalid_view_tile_size.bc -no-implicit-module 2>&1 | FileCheck %s --check-prefix=INVALID_VIEW_TILE_SIZE
// INVALID_VIEW_TILE_SIZE: tile would exceed the maximum of 16777216 elements
// INVALID_VIEW_TILE_SIZE: failed to get result type 0 for MakePartitionViewOp
// INVALID_VIEW_TILE_SIZE: failed to parse function body for function 'kernel'
// INVALID_VIEW_TILE_SIZE: failed to create function from bytecode
