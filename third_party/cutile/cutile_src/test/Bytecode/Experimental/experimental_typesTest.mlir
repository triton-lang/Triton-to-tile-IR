// RUN: %round_trip_test %s %t


cuda_tile.module @kernels {
  cuda_tile.experimental$func @scalar_i32_type(%b: !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    %0 = cuda_tile.addi %b, %b : !cuda_tile.tile<i32>
    cuda_tile.return %0 : tile<i32>
  }

  cuda_tile.experimental$func @nested_func(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) -> !cuda_tile.tile<i32> {
    %0 = cuda_tile.addi %a, %b : !cuda_tile.tile<i32>
    cuda_tile.return %0 : tile<i32>
  }

  cuda_tile.experimental$func @scalar_i64_type(%b: !cuda_tile.tile<i64>) -> !cuda_tile.tile<i64> {
    %0 = cuda_tile.addi %b, %b : !cuda_tile.tile<i64>
    cuda_tile.return %0 : tile<i64>
  }

  // Integer types
  cuda_tile.experimental$func @scalar_i1_type(%b: !cuda_tile.tile<i1>) -> !cuda_tile.tile<i1> {
    cuda_tile.return %b : tile<i1>
  }
  
  cuda_tile.experimental$func @scalar_i8_type(%b: !cuda_tile.tile<i8>) -> !cuda_tile.tile<i8> {
    cuda_tile.return %b : tile<i8>
  }
  
  cuda_tile.experimental$func @scalar_i16_type(%b: !cuda_tile.tile<i16>) -> !cuda_tile.tile<i16> {
    cuda_tile.return %b : tile<i16>
  }
  
  // Float types
  cuda_tile.experimental$func @scalar_f32_type(%b: !cuda_tile.tile<f32>) -> !cuda_tile.tile<f32> {
    cuda_tile.return %b : tile<f32>
  }
  
  cuda_tile.experimental$func @scalar_f16_type(%b: !cuda_tile.tile<f16>) -> !cuda_tile.tile<f16> {
    cuda_tile.return %b : tile<f16>
  }
  
  cuda_tile.experimental$func @scalar_bf16_type(%b: !cuda_tile.tile<bf16>) -> !cuda_tile.tile<bf16> {
    cuda_tile.return %b : tile<bf16>
  }
  
  cuda_tile.experimental$func @scalar_f64_type(%b: !cuda_tile.tile<f64>) -> !cuda_tile.tile<f64> {
    cuda_tile.return %b : tile<f64>
  }
  
  cuda_tile.experimental$func @scalar_f8e4m3fn_type(%b: !cuda_tile.tile<f8E4M3FN>) -> !cuda_tile.tile<f8E4M3FN> {
    cuda_tile.return %b : tile<f8E4M3FN>
  }
  
  cuda_tile.experimental$func @scalar_f8e5m2_type(%b: !cuda_tile.tile<f8E5M2>) -> !cuda_tile.tile<f8E5M2> {
    cuda_tile.return %b : tile<f8E5M2>
  }
    
  cuda_tile.experimental$func @scalar_tf32_type(%b: !cuda_tile.tile<tf32>) -> !cuda_tile.tile<tf32> {
    cuda_tile.return %b : tile<tf32>
  }
  
  // Multidimensional tensor
  cuda_tile.experimental$func @multidim_tensor_type(%b: !cuda_tile.tile<2x4xi32>) -> !cuda_tile.tile<2x4xi32> {
    cuda_tile.return %b : tile<2x4xi32>
  }
  
  // Pointer type
  cuda_tile.experimental$func @pointer_type(%b: !cuda_tile.ptr<i32>) -> !cuda_tile.ptr<i32> {
    cuda_tile.return %b : ptr<i32>
  }
  
  // TensorView type
  cuda_tile.experimental$func @memref_type(%b: !cuda_tile.tensor_view<16x16xf32, strides=[16,1]>) -> !cuda_tile.tensor_view<16x16xf32, strides=[16,1]> {
    cuda_tile.return %b : tensor_view<16x16xf32, strides=[16,1]>
  }
  
  // TensorView with dynamic dimensions
  cuda_tile.experimental$func @memref_dynamic_type(%b: !cuda_tile.tensor_view<?x?xf32, strides=[?,1]>) -> !cuda_tile.tensor_view<?x?xf32, strides=[?,1]> {
    cuda_tile.return %b : tensor_view<?x?xf32, strides=[?,1]>
  }
  
  // TensorView with 64 bit bitwidth
  cuda_tile.experimental$func @memref_index_type(%b: !cuda_tile.tensor_view<16xf32, strides=[1]>) -> !cuda_tile.tensor_view<16xf32, strides=[1]> {
    cuda_tile.return %b : tensor_view<16xf32, strides=[1]>
  }
  
  // PartitionView type
  cuda_tile.experimental$func @partition_view_type(%b: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<16x16xf32, strides=[16,1]>>) -> !cuda_tile.partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>> {
    cuda_tile.return %b : partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>>
  }

  // PartitionView with dimension map
  cuda_tile.experimental$func @tile_partition_dim_map_type(%b: !cuda_tile.partition_view<tile=(2x2), !cuda_tile.tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]>) -> !cuda_tile.partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]> {
    cuda_tile.return %b : partition_view<tile=(2x2), tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]>
  }

  // StridedView type
  cuda_tile.experimental$func @strided_view_type(%b: !cuda_tile.strided_view<tile=(2x2), traversal_strides=[1,1], !cuda_tile.tensor_view<16x16xf32, strides=[16,1]>>) -> !cuda_tile.strided_view<tile=(2x2), traversal_strides=[1,1], tensor_view<16x16xf32, strides=[16,1]>> {
    cuda_tile.return %b : strided_view<tile=(2x2), traversal_strides=[1,1], tensor_view<16x16xf32, strides=[16,1]>>
  }

  // StridedView with dimension map
  cuda_tile.experimental$func @tile_strided_view_dim_map_type(%b: !cuda_tile.strided_view<tile=(2x2), traversal_strides=[5,1], !cuda_tile.tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]>) -> !cuda_tile.strided_view<tile=(2x2), traversal_strides=[5,1], tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]> {
    cuda_tile.return %b : strided_view<tile=(2x2), traversal_strides=[5,1], tensor_view<16x16xf32, strides=[16,1]>, dim_map=[1, 0]>
  }

  cuda_tile.experimental$func @token_type(%tok: !cuda_tile.token) -> !cuda_tile.token {
    cuda_tile.return %tok : token
  }
}
