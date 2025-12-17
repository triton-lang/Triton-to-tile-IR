// RUN: cuda-tile-opt --inline -split-input-file %s | FileCheck %s

cuda_tile.module @test_direct_call_op {

experimental$func @direct_callsite_fn(%arg0 : !cuda_tile.tile<2xi16>) {
}

// CHECK-LABEL: @kernel
experimental$func @kernel(%arg0: !cuda_tile.tile<2xi16>) {
  // CHECK-NOT: experimental$call
  experimental$call @direct_callsite_fn(%arg0) : (!cuda_tile.tile<2xi16>) -> ()
}
}

// -----

cuda_tile.module @test_direct_call_op {

experimental$func @direct_callsite_fn(%arg0 : !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16> {
  return %arg0 : tile<2xi16>
}

// CHECK-LABEL: @kernel
// CHECK-SAME: %[[ARG0:.+]]: tile<2xi16>
experimental$func @kernel(%arg0: !cuda_tile.tile<2xi16>) ->  !cuda_tile.tile<2xi16> {
  // CHECK-NOT: experimental$call
  // CHECK: return %[[ARG0]] : tile<2xi16>
  %0 = experimental$call @direct_callsite_fn(%arg0) : (!cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16>
  return %0 : tile<2xi16>
}
}

// -----

cuda_tile.module @test_direct_call_op {

experimental$func @direct_callsite_fn(%arg0 : !cuda_tile.tile<2xi16>, %arg1 : !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16> {
  %0 = addi %arg0, %arg1 : tile<2xi16>
  return %0 : tile<2xi16>
}

// CHECK-LABEL: @kernel
// CHECK-SAME:  %[[ARG0:.+]]: tile<2xi16>,
// CHECK-SAME:  %[[ARG1:.+]]: tile<2xi16>
experimental$func @kernel(%arg0: !cuda_tile.tile<2xi16>, %arg1: !cuda_tile.tile<2xi16>) ->  !cuda_tile.tile<2xi16> {
  // CHECK-NOT: experimental$call
  // CHECK: %[[ADD:.+]] = addi %[[ARG0]], %[[ARG1]] : tile<2xi16>
  // CHECK-NEXT: return %[[ADD]] : tile<2xi16>
  %0 = experimental$call @direct_callsite_fn(%arg0, %arg1) : (!cuda_tile.tile<2xi16>, !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16>
  return %0 : tile<2xi16>
}
}

// -----

cuda_tile.module @test_direct_call_op {

experimental$func @direct_callsite_fn_single(%arg0 : !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16> {
  return %arg0 : tile<2xi16>
}

experimental$func @direct_callsite_fn(%arg0 : !cuda_tile.tile<2xi16>, %arg1 : !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16> {
  %0 = experimental$call @direct_callsite_fn_single(%arg0) : (!cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16>
  return %0 : tile<2xi16>
}

// CHECK-LABEL: inlinable_kernel(
// CHECK-SAME: %[[ARG0:.+]]: tile<2xi16>,
experimental$func @inlinable_kernel(%arg0: !cuda_tile.tile<2xi16>, %arg1: !cuda_tile.tile<2xi16>) ->  !cuda_tile.tile<2xi16> {
  // CHECK-NOT: experimental$call
  // CHECK: return %[[ARG0]] : tile<2xi16>
  %0 = experimental$call @direct_callsite_fn(%arg0, %arg1) : (!cuda_tile.tile<2xi16>, !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16>
  return %0 : tile<2xi16>
}
}

// -----

cuda_tile.module @test_early_exit {
experimental$func @early_exit_fn(%cond: !cuda_tile.tile<i1>, %arg0 : !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16> {
  if %cond {
    %0 = addi %arg0, %arg0 : tile<2xi16>
    return %0 : tile<2xi16>
  }
  return %arg0 : tile<2xi16>
}

// CHECK-LABEL: early_exit_kernel
// CHECK-SAME: %[[COND:.+]]: tile<i1>,
// CHECK-SAME: %[[ARG0:.+]]: tile<2xi16>
experimental$func @early_exit_kernel(%cond: !cuda_tile.tile<i1>, %arg0 : !cuda_tile.tile<2xi16>) ->  !cuda_tile.tile<2xi16> {
  // Check that in the face of early exit control flow, we wrap with a loop to still model the early exit.
  // CHECK: %[[RESULT:.+]] = loop : tile<2xi16>
  // CHECK:   if %[[COND]] {
  // CHECK:     %[[ADD:.+]] = addi %[[ARG0]], %[[ARG0]] : tile<2xi16>
  // CHECK:     break %[[ADD]] : tile<2xi16>
  // CHECK:   }
  // CHECK:   break %[[ARG0]] : tile<2xi16>
  // CHECK: }
  // CHECK: return %[[RESULT]] : tile<2xi16>
  %0 = experimental$call @early_exit_fn(%cond, %arg0) : (!cuda_tile.tile<i1>, !cuda_tile.tile<2xi16>) -> !cuda_tile.tile<2xi16>
  return %0 : tile<2xi16>
}
}
