// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// expected-error @below{{'cuda_tile.experimental$generate' op invalid yielded value from the region: operand has null or non-tile type}}
%0 = "cuda_tile.experimental$generate"() ({
^bb0(%arg0: !cuda_tile.tile<i32>, %arg1: !cuda_tile.tile<i32>):
  %1 = "cuda_tile.subi"(%arg1, %arg0) <{overflow = #cuda_tile.overflow<none>}> : (!cuda_tile.tile<i32>, !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32>
  %2 = "cuda_tile.addi"(%1, %arg0) <{overflow = #cuda_tile.overflow<none>}> : (!cuda_tile.tile<i32>, !cuda_tile.tile<i32>) -> i32
  "cuda_tile.yield"(%2) : (i32) -> ()
}) : () -> !cuda_tile.tile<16x16xi32>

// -----

cuda_tile.testing$func @test_invalid_early_return(%arg0: !cuda_tile.tile<2x!cuda_tile.ptr<i32>>,
                                          %arg1: !cuda_tile.tile<2xi32>) {
  %c1 = constant <i1: true> : !cuda_tile.tile<i1>
  loop {
    if %c1 {
      // expected-error @below {{must be used within a cuda_tile.experimental$func, cuda_tile.testing$func, cuda_tile.entry, or cuda_tile.if operation}}
      return
    }
  }
}
