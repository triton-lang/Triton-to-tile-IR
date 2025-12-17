// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s

cuda_tile.module @kernels {

// CHECK-LABEL: experimental$func @test_fence
experimental$func @test_fence() {
  // CHECK: internal$fence_ordered
  internal$fence_ordered relaxed device
  return
}

// CHECK-LABEL: test_not_after_ordered(
// CHECK-SAME: %[[tok:.+]]: token
experimental$func @test_not_after_ordered(%token: !cuda_tile.token) {
  // CHECK: internal$not_after_ordered %[[tok]]
  internal$not_after_ordered %token
  return
}

// CHECK-LABEL: test_not_before_ordered()
experimental$func @test_not_before_ordered() {
  // CHECK: %{{.+}} = internal$not_before_ordered : token
  %0 = internal$not_before_ordered : token
  return
}

// CHECK-LABEL: ordered_load
experimental$func @ordered_load(%ptr: !cuda_tile.tile<16x32xptr<f32>>) {
  // CHECK: internal$load_ordered weak %{{.+}} : tile<16x32xptr<f32>> -> tile<16x32xf32>
  %0 = internal$load_ordered weak %ptr : tile<16x32xptr<f32>> -> tile<16x32xf32>
  return
}

// CHECK-LABEL: ordered_load_device
experimental$func @ordered_load_device(%ptr: !cuda_tile.tile<16x32xptr<f32>>) {
  // CHECK: internal$load_ordered acquire device %{{.+}} : tile<16x32xptr<f32>> -> tile<16x32xf32>
  %0 = internal$load_ordered acquire device %ptr : tile<16x32xptr<f32>> -> tile<16x32xf32>
  return
}

} // end module