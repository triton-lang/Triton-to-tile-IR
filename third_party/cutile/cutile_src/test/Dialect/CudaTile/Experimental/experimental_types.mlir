// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s

cuda_tile.module @kernels {

// CHECK-LABEL: testing$func @test_strided_view_types
testing$func @test_strided_view_types(
    // CHECK-SAME: strided_view<tile=(2), traversal_strides=[1], tensor_view<16xf32, strides=[1]>>
    %arg0: !cuda_tile.strided_view<tile=(2), traversal_strides=[1], tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: strided_view<tile=(2), traversal_strides=[1], padding_value = zero, tensor_view<16xf32, strides=[1]>>
    %arg1: !cuda_tile.strided_view<tile=(2), traversal_strides=[1], padding_value = zero, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: strided_view<tile=(2), traversal_strides=[2], padding_value = nan, tensor_view<16xf32, strides=[1]>>
    %arg2: !cuda_tile.strided_view<tile=(2), traversal_strides=[2], padding_value = nan, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: strided_view<tile=(2), traversal_strides=[2], padding_value = neg_zero, tensor_view<16xf32, strides=[1]>>
    %arg3: !cuda_tile.strided_view<tile=(2), traversal_strides=[2], padding_value = neg_zero, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: strided_view<tile=(2), traversal_strides=[5], padding_value = pos_inf, tensor_view<16xf32, strides=[1]>>
    %arg4: !cuda_tile.strided_view<tile=(2), traversal_strides=[5], padding_value = pos_inf, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: strided_view<tile=(2), traversal_strides=[5], padding_value = neg_inf, tensor_view<16xf32, strides=[1]>>
    %arg5: !cuda_tile.strided_view<tile=(2), traversal_strides=[5], padding_value = neg_inf, tensor_view<16xf32, strides=[1]>>,
    // CHECK-SAME: strided_view<tile=(2), traversal_strides=[100], tensor_view<16xf32, strides=[1]>>
    %arg6: !cuda_tile.strided_view<tile=(2), traversal_strides=[100], tensor_view<16xf32, strides=[1]>, dim_map=[0]>,
    // CHECK-SAME: strided_view<tile=(2x2), traversal_strides=[8,1], tensor_view<16x16xf32, strides=[16,1]>>
    %arg7: !cuda_tile.strided_view<tile=(2x2), traversal_strides=[8,1], tensor_view<16x16xf32, strides=[16,1]>>,
    // CHECK-SAME: strided_view<tile=(2x2), traversal_strides=[8,8], tensor_view<16x16xf32, strides=[16,1]>>
    %arg8: !cuda_tile.strided_view<tile=(2x2), traversal_strides=[8,8], tensor_view<16x16xf32, strides=[16,1]>, dim_map=[0, 1]>,
    // CHECK-SAME: strided_view<tile=(2x2), traversal_strides=[7,6], tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]>
    %arg9: !cuda_tile.strided_view<tile=(2x2), traversal_strides=[7,6], tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]>) {
  return
}
}