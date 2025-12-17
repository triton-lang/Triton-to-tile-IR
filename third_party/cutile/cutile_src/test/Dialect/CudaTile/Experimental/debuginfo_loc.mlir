// RUN: cuda-tile-opt --mlir-print-debuginfo %s | FileCheck %s
// RUN: %S/../round_trip_test.sh %s %t --mlir-print-debuginfo

#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>

//===----------------------------------------------------------------------===//
// Shared base locations for entry tests
//===----------------------------------------------------------------------===//

// CHECK-DAG: [[UNKNOWN:#loc[0-9]*]] = loc(unknown)
#unknown = loc(unknown)

// CHECK-DAG: [[LOC:#loc[0-9]*]] = loc("/tmp/foo.py":3:4)
#loc = loc("/tmp/foo.py":3:4)

//===----------------------------------------------------------------------===//
// Entry function debug info
//===----------------------------------------------------------------------===//
#entry_subprogram = #cuda_tile.di_subprogram<
  file = #file,
  line = 10,
  name = "test_locations_entry",
  linkageName = "test_locations_entry",
  compileUnit = #compile_unit,
  scopeLine = 11
>

#entry_lexical_block = #cuda_tile.di_lexical_block<
  scope = #entry_subprogram,
  file = #file,
  line = 15,
  column = 8
>

#entry_inner_lexical_block = #cuda_tile.di_lexical_block<
  scope = #entry_lexical_block,
  file = #file,
  line = 17,
  column = 12
>

// Entry function location definitions
// CHECK-DAG: [[ENTRY_DI_LOC:#loc[0-9]*]] = #cuda_tile.di_loc<[[LOC]] in #di_subprogram>
#entry_di_loc = #cuda_tile.di_loc<#loc in #entry_subprogram>

// CHECK-DAG: [[ENTRY_BLOCK_LOC:#loc[0-9]*]] = loc("/tmp/foo.py":15:8)
// CHECK-DAG: [[ENTRY_BLOCK_DI_LOC:#loc[0-9]*]] = #cuda_tile.di_loc<[[ENTRY_BLOCK_LOC]] in #di_lexical_block>
#entry_block_loc = loc("/tmp/foo.py":15:8)
#entry_block_di_loc = #cuda_tile.di_loc<#entry_block_loc in #entry_lexical_block>

// CHECK-DAG: [[ENTRY_INNER_BLOCK_LOC:#loc[0-9]*]] = loc("/tmp/foo.py":17:12)
// CHECK-DAG: [[ENTRY_INNER_BLOCK_DI_LOC:#loc[0-9]*]] = #cuda_tile.di_loc<[[ENTRY_INNER_BLOCK_LOC]] in #di_lexical_block1>
#entry_inner_block_loc = loc("/tmp/foo.py":17:12)
#entry_inner_block_di_loc = #cuda_tile.di_loc<#entry_inner_block_loc in #entry_inner_lexical_block>

// Entry function callsite locations
// CHECK-DAG: [[ENTRY_CALLSITE1:#loc[0-9]*]] = loc(callsite([[ENTRY_DI_LOC]] at [[ENTRY_DI_LOC]]))
// CHECK-DAG: [[ENTRY_CALLSITE_BLOCK1:#loc[0-9]*]] = loc(callsite([[ENTRY_BLOCK_DI_LOC]] at [[ENTRY_DI_LOC]]))
// CHECK-DAG: [[ENTRY_CALLSITE_BLOCK2:#loc[0-9]*]] = loc(callsite([[ENTRY_INNER_BLOCK_DI_LOC]] at [[ENTRY_BLOCK_DI_LOC]]))
#entry_callsite1 = loc(callsite(#entry_di_loc at #entry_di_loc))
#entry_callsite_block1 = loc(callsite(#entry_block_di_loc at #entry_di_loc))
#entry_callsite_block2 = loc(callsite(#entry_inner_block_di_loc at #entry_block_di_loc))

// Entry function callsite chains
// CHECK-DAG: [[ENTRY_CALLSITE_CHAIN1:#loc[0-9]*]] = loc(callsite([[ENTRY_CALLSITE1]] at [[ENTRY_DI_LOC]]))
// CHECK-DAG: [[ENTRY_CALLSITE_CHAIN2:#loc[0-9]*]] = loc(callsite([[ENTRY_DI_LOC]] at [[ENTRY_CALLSITE1]]))
// CHECK-DAG: [[ENTRY_CALLSITE_CHAIN3:#loc[0-9]*]] = loc(callsite([[ENTRY_CALLSITE_BLOCK1]] at [[ENTRY_CALLSITE_BLOCK2]]))
// CHECK-DAG: [[ENTRY_CALLSITE_CHAIN4:#loc[0-9]*]] = loc(callsite([[ENTRY_CALLSITE1]] at [[ENTRY_CALLSITE1]]))
// CHECK-DAG: [[ENTRY_CALLSITE_CHAIN5:#loc[0-9]*]] = loc(callsite([[ENTRY_CALLSITE_CHAIN4]] at [[UNKNOWN]]))
#entry_callsite_chain1 = loc(callsite(#entry_callsite1 at #entry_di_loc))
#entry_callsite_chain2 = loc(callsite(#entry_di_loc at #entry_callsite1))
#entry_callsite_chain3 = loc(callsite(#entry_callsite_block1 at #entry_callsite_block2))
#entry_callsite_chain4 = loc(callsite(#entry_callsite1 at #entry_callsite1))
#entry_callsite_chain5 = loc(callsite(#entry_callsite_chain4 at #unknown))
//===----------------------------------------------------------------------===//
// Experimental function tests
//===----------------------------------------------------------------------===//

cuda_tile.module @kernels {
  // Global variables - Test Rule 5: Global variables must not have scope
  // These use location wrappers but only with non-scoped locations (UnknownLoc, FileLineColLoc)

  // CHECK-DAG: global @global_unknown <i32: 0> : tile<1xi32> loc([[UNKNOWN]])
  global @global_unknown <i32: [0]> : !cuda_tile.tile<1xi32> loc(#unknown)

//===----------------------------------------------------------------------===//
// Entry function tests
//===----------------------------------------------------------------------===//

  // CHECK-DAG: entry @test_locations_entry()
  // CHECK-DAG:   constant <i32: 100> : tile<i32> loc([[UNKNOWN]])
  // CHECK-DAG:   constant <i32: 102> : tile<i32> loc([[ENTRY_DI_LOC]])
  // CHECK-DAG:   constant <i32: 107> : tile<i32> loc([[ENTRY_CALLSITE1]])
  // CHECK-DAG:   constant <i32: 111> : tile<i32> loc([[ENTRY_BLOCK_DI_LOC]])
  // CHECK-DAG:   constant <i32: 112> : tile<i32> loc([[ENTRY_INNER_BLOCK_DI_LOC]])
  // CHECK-DAG:   constant <i32: 117> : tile<i32> loc([[ENTRY_CALLSITE_BLOCK1]])
  // CHECK-DAG:   constant <i32: 118> : tile<i32> loc([[ENTRY_CALLSITE_BLOCK2]])
  // CHECK-DAG:   constant <i32: 120> : tile<i32> loc([[ENTRY_CALLSITE_CHAIN1]])
  // CHECK-DAG:   constant <i32: 121> : tile<i32> loc([[ENTRY_CALLSITE_CHAIN2]])
  // CHECK-DAG:   constant <i32: 122> : tile<i32> loc([[ENTRY_CALLSITE_CHAIN3]])
  // CHECK-DAG:   constant <i32: 123> : tile<i32> loc([[ENTRY_CALLSITE_CHAIN4]])
  // CHECK-DAG:   constant <i32: 124> : tile<i32> loc([[ENTRY_CALLSITE_CHAIN5]])
  // CHECK-DAG: } loc([[ENTRY_DI_LOC]])
  entry @test_locations_entry() {
    %c100 = constant <i32: 100> : !cuda_tile.tile<i32> loc(#unknown)
    %c102 = constant <i32: 102> : !cuda_tile.tile<i32> loc(#entry_di_loc)
    %c107 = constant <i32: 107> : !cuda_tile.tile<i32> loc(#entry_callsite1)
    %c111 = constant <i32: 111> : !cuda_tile.tile<i32> loc(#entry_block_di_loc)
    %c112 = constant <i32: 112> : !cuda_tile.tile<i32> loc(#entry_inner_block_di_loc)
    %c117 = constant <i32: 117> : !cuda_tile.tile<i32> loc(#entry_callsite_block1)
    %c118 = constant <i32: 118> : !cuda_tile.tile<i32> loc(#entry_callsite_block2)
    %c120 = constant <i32: 120> : !cuda_tile.tile<i32> loc(#entry_callsite_chain1)
    %c121 = constant <i32: 121> : !cuda_tile.tile<i32> loc(#entry_callsite_chain2)
    %c122 = constant <i32: 122> : !cuda_tile.tile<i32> loc(#entry_callsite_chain3)
    %c123 = constant <i32: 123> : !cuda_tile.tile<i32> loc(#entry_callsite_chain4)
    %c124 = constant <i32: 124> : !cuda_tile.tile<i32> loc(#entry_callsite_chain5)
    return loc(#unknown)
  } loc(#entry_di_loc)
} loc(unknown)
