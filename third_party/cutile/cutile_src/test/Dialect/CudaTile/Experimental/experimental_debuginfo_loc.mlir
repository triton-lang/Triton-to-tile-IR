// RUN: cuda-tile-opt --mlir-print-debuginfo %s | FileCheck %s
// RUN: %S/../round_trip_test.sh %s %t --mlir-print-debuginfo

#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>

//===----------------------------------------------------------------------===//
// Experimental function debug info
//===----------------------------------------------------------------------===//

#subprogram = #cuda_tile.di_subprogram<
  file = #file,
  line = 1,
  name = "test_locations",
  linkageName = "test_locations",
  compileUnit = #compile_unit,
  scopeLine = 2
>

#lexical_block = #cuda_tile.di_lexical_block<
  scope = #subprogram,
  file = #file,
  line = 5,
  column = 8
>

#inner_lexical_block = #cuda_tile.di_lexical_block<
  scope = #lexical_block,
  file = #file,
  line = 7,
  column = 12
>

// CHECK-DAG: [[UNKNOWN:#loc[0-9]*]] = loc(unknown)
#unknown = loc(unknown)

// CHECK-DAG: [[LOC:#loc[0-9]*]] = loc("/tmp/foo.py":3:4)
// CHECK-DAG: [[DI_LOC:#loc[0-9]*]] = #cuda_tile.di_loc<[[LOC]] in #di_subprogram>
#loc = loc("/tmp/foo.py":3:4)
#di_loc = #cuda_tile.di_loc<#loc in #subprogram>

// CHECK-DAG: [[BLOCK_LOC:#loc[0-9]*]] = loc("/tmp/foo.py":5:8)
// CHECK-DAG: [[BLOCK_DI_LOC:#loc[0-9]*]] = #cuda_tile.di_loc<[[BLOCK_LOC]] in #di_lexical_block>
#block_loc = loc("/tmp/foo.py":5:8)
#block_di_loc = #cuda_tile.di_loc<#block_loc in #lexical_block>

// CHECK-DAG: [[INNER_BLOCK_LOC:#loc[0-9]*]] = loc("/tmp/foo.py":7:12)
// CHECK-DAG: [[INNER_BLOCK_DI_LOC:#loc[0-9]*]] = #cuda_tile.di_loc<[[INNER_BLOCK_LOC]] in #di_lexical_block1>
#inner_block_loc = loc("/tmp/foo.py":7:12)
#inner_block_di_loc = #cuda_tile.di_loc<#inner_block_loc in #inner_lexical_block>


// CHECK-DAG: [[CALLSITE1:#loc[0-9]*]] = loc(callsite([[DI_LOC]] at [[DI_LOC]]))
// CHECK-DAG: [[CALLSITE_BLOCK1:#loc[0-9]*]] = loc(callsite([[BLOCK_DI_LOC]] at [[DI_LOC]]))
// CHECK-DAG: [[CALLSITE_BLOCK2:#loc[0-9]*]] = loc(callsite([[INNER_BLOCK_DI_LOC]] at [[BLOCK_DI_LOC]]))
#callsite1 = loc(callsite(#di_loc at #di_loc))
#callsite_block1 = loc(callsite(#block_di_loc at #di_loc))
#callsite_block2 = loc(callsite(#inner_block_di_loc at #block_di_loc))

// CallSiteLoc chains - testing CallSiteLoc where caller/callee are themselves CallSiteLoc
// CHECK-DAG: [[CALLSITE_CHAIN1:#loc[0-9]*]] = loc(callsite([[CALLSITE1]] at [[DI_LOC]]))
// CHECK-DAG: [[CALLSITE_CHAIN2:#loc[0-9]*]] = loc(callsite([[DI_LOC]] at [[CALLSITE1]]))
// CHECK-DAG: [[CALLSITE_CHAIN3:#loc[0-9]*]] = loc(callsite([[CALLSITE_BLOCK1]] at [[CALLSITE_BLOCK2]]))
// CHECK-DAG: [[CALLSITE_CHAIN4:#loc[0-9]*]] = loc(callsite([[CALLSITE1]] at [[CALLSITE1]]))
// CHECK-DAG: [[CALLSITE_CHAIN5:#loc[0-9]*]] = loc(callsite([[CALLSITE_CHAIN4]] at [[UNKNOWN]]))
#callsite_chain1 = loc(callsite(#callsite1 at #di_loc))
#callsite_chain2 = loc(callsite(#di_loc at #callsite1))
#callsite_chain3 = loc(callsite(#callsite_block1 at #callsite_block2))
#callsite_chain4 = loc(callsite(#callsite1 at #callsite1))
#callsite_chain5 = loc(callsite(#callsite_chain4 at #unknown))


//===----------------------------------------------------------------------===//
// Experimental function tests
//===----------------------------------------------------------------------===//

cuda_tile.module @kernels {
  // CHECK-DAG: experimental$func @test_locations()
  // CHECK-DAG:   constant <i32: 0> : tile<i32> loc([[UNKNOWN]])
  // CHECK-DAG:   constant <i32: 2> : tile<i32> loc([[DI_LOC]])
  // CHECK-DAG:   constant <i32: 7> : tile<i32> loc([[CALLSITE1]])
  // CHECK-DAG:   constant <i32: 11> : tile<i32> loc([[BLOCK_DI_LOC]])
  // CHECK-DAG:   constant <i32: 12> : tile<i32> loc([[INNER_BLOCK_DI_LOC]])
  // CHECK-DAG:   constant <i32: 17> : tile<i32> loc([[CALLSITE_BLOCK1]])
  // CHECK-DAG:   constant <i32: 18> : tile<i32> loc([[CALLSITE_BLOCK2]])
  // CHECK-DAG:   constant <i32: 20> : tile<i32> loc([[CALLSITE_CHAIN1]])
  // CHECK-DAG:   constant <i32: 21> : tile<i32> loc([[CALLSITE_CHAIN2]])
  // CHECK-DAG:   constant <i32: 22> : tile<i32> loc([[CALLSITE_CHAIN3]])
  // CHECK-DAG:   constant <i32: 23> : tile<i32> loc([[CALLSITE_CHAIN4]])
  // CHECK-DAG:   constant <i32: 24> : tile<i32> loc([[CALLSITE_CHAIN5]])
  // CHECK-DAG: } loc([[DI_LOC]])
  experimental$func @test_locations() {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32> loc(#unknown)
    %c2 = constant <i32: 2> : !cuda_tile.tile<i32> loc(#di_loc)
    %c7 = constant <i32: 7> : !cuda_tile.tile<i32> loc(#callsite1)
    %c11 = constant <i32: 11> : !cuda_tile.tile<i32> loc(#block_di_loc)
    %c12 = constant <i32: 12> : !cuda_tile.tile<i32> loc(#inner_block_di_loc)
    %c17 = constant <i32: 17> : !cuda_tile.tile<i32> loc(#callsite_block1)
    %c18 = constant <i32: 18> : !cuda_tile.tile<i32> loc(#callsite_block2)
    %c20 = constant <i32: 20> : !cuda_tile.tile<i32> loc(#callsite_chain1)
    %c21 = constant <i32: 21> : !cuda_tile.tile<i32> loc(#callsite_chain2)
    %c22 = constant <i32: 22> : !cuda_tile.tile<i32> loc(#callsite_chain3)
    %c23 = constant <i32: 23> : !cuda_tile.tile<i32> loc(#callsite_chain4)
    %c24 = constant <i32: 24> : !cuda_tile.tile<i32> loc(#callsite_chain5)
    return loc(#unknown)
  } loc(#di_loc)

  // CHECK-DAG: experimental$func @unknown()
  // CHECK-DAG:   constant <i32: 100> : tile<i32> loc([[UNKNOWN]])
  // CHECK-DAG: } loc([[UNKNOWN]])
  experimental$func @unknown() {
    %c100 = constant <i32: 100> : !cuda_tile.tile<i32> loc(#unknown)
    return loc(#unknown)
  } loc(#unknown)


} loc(unknown)
