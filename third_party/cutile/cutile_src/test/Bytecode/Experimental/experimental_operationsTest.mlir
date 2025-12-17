// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  // Test permute operation with multidimensional tiles
  cuda_tile.experimental$func @permute_op(%a: !cuda_tile.tile<2x4x8xf32>) {
    %0 = cuda_tile.permute %a [2, 0, 1] : tile<2x4x8xf32> -> tile<8x2x4xf32>
    cuda_tile.return
  }

  // Test reduce operation with 2D tiles
  cuda_tile.experimental$func @reduce_operation_2d_dim1(%arg0: !cuda_tile.tile<8x64xf32>) {
    %0 = cuda_tile.reduce %arg0 dim=1 identities=[0.0 : f32] : !cuda_tile.tile<8x64xf32> -> !cuda_tile.tile<8xf32>
    (%arg0_in : !cuda_tile.tile<f32>, %arg0_identity : !cuda_tile.tile<f32>) {
      %add = cuda_tile.addf %arg0_in, %arg0_identity rounding<nearest_even> : !cuda_tile.tile<f32>
      cuda_tile.yield %add : !cuda_tile.tile<f32>
    }
    cuda_tile.return
  }

  // Test scan operation with 2D tiles
  cuda_tile.experimental$func @scan_operation_2d_dim0(%arg0: !cuda_tile.tile<8x64xf32>) {
    %0 = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.0 : f32] : !cuda_tile.tile<8x64xf32> -> !cuda_tile.tile<8x64xf32>
      (%arg0_in : !cuda_tile.tile<f32>, %arg0_identity : !cuda_tile.tile<f32>) {
        %add = cuda_tile.addf %arg0_in, %arg0_identity rounding<nearest_even> : !cuda_tile.tile<f32>
        cuda_tile.yield %add : !cuda_tile.tile<f32>
      }
    cuda_tile.return
  }

  // Test load/store with partition views and multidimensional tiles
  experimental$func @load_store_partition_view(%view1: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>,
                                               %view3: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>,
                                               %t1: !cuda_tile.tile<8xf32>, %t3: !cuda_tile.tile<1024x1024x8xf32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
  
    %s1 = store_view_tko weak %t1, %view1[%c0] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    %s2 = store_view_tko weak %t3, %view3[%c0, %c0, %c0] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token
    %s3 = store_view_tko weak %t3, %view3[%c0, %c0, %c0] : tile<1024x1024x8xf32>, partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token

    %t1_l, %tok0 = load_view_tko weak %view1[%c0] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    %t2_l, %tok1 = load_view_tko weak %view1[%c0] : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    %t3_l, %tok2 = load_view_tko weak %view3[%c0, %c0, %c0] : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
  }

  // Test load/store with strided views and multidimensional tiles
  experimental$func @load_store_strided_view(%view1: !cuda_tile.strided_view<tile=(8), traversal_strides=[1], !cuda_tile.tensor_view<128xf32, strides=[1]>>,
                                             %view3: !cuda_tile.strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>,
                                             %t1: !cuda_tile.tile<8xf32>, %t3: !cuda_tile.tile<1024x1024x8xf32>) {
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
  
    %s1 = store_view_tko weak %t1, %view1[%c0] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    %s2 = store_view_tko weak %t3, %view3[%c0, %c0, %c0] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token
    %s3 = store_view_tko weak %t3, %view3[%c0, %c0, %c0] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token

    %t1_l, %tok0 = load_view_tko weak %view1[%c0] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i32> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    %t2_l, %tok1 = load_view_tko weak %view1[%c0] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i32> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
    %t3_l, %tok2 = load_view_tko weak %view3[%c0, %c0, %c0] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
  }

  // Test atomic operations with array tiles
  experimental$func @atomic_rmw_tko(%arg0: !cuda_tile.tile<2xptr<i32>>,
                                     %arg1: !cuda_tile.tile<2xi32>,
                                     %arg2: !cuda_tile.tile<2xptr<f32>>) {
    %0, %t = atomic_rmw_tko relaxed device %arg0, and, %arg1
        : tile<2xptr<i32>>, tile<2xi32> -> tile<2xi32>, token
  }

  experimental$func @pack_op(%arg0: !cuda_tile.tile<32xf16>) {
    %0 = experimental$pack %arg0 : tile<32xf16> -> tile<64xi8>
    return
  }

  experimental$func @unpack_op(%arg0: !cuda_tile.tile<64xi8>) {
    %0 = experimental$unpack %arg0 : tile<64xi8> -> tile<128xf4E2M1FN>
    return
  }
}
