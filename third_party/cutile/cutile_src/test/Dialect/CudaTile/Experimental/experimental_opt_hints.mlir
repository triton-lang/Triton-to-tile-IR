// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt | FileCheck %s
// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  // CHECK-LABEL: tiled_view_load_allow_tma
  // TODO: Bring attributes {allow_tma = false} once support added.
  experimental$func @tiled_view_load_allow_tma(%arg0: !cuda_tile.partition_view<tile=(1024x1024x8), !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, %arg1: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>

    // CHECK: %{{.+}}, %{{.+}} = load_view_tko weak %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] optimization_hints = <sm_100 = {latency = 5}> : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32>
    // CHECK-SAME:  -> tile<1024x1024x8xf32>, token
    %tile_3, %tok_out_1 = load_view_tko weak %arg0[%0, %0, %0] optimization_hints = <sm_100={latency = 5}> : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token

    // CHECK: %{{.+}}, %{{.+}} = load_view_tko relaxed device %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}] token = %{{.+}} optimization_hints = <sm_100 = {allow_tma = false}> : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32>
    // CHECK-SAME: -> tile<1024x1024x8xf32>, token
    %tile_4, %tok_out_2 = load_view_tko relaxed device %arg0[%0, %0, %0] token = %arg1 optimization_hints = <sm_100={allow_tma = false}> : partition_view<tile=(1024x1024x8), tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
    return
  }

  // CHECK-LABEL: tiled_view_store_allow_tma
  // TODO: Bring attributes {allow_tma = false} once support added.
  experimental$func @tiled_view_store_allow_tma(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.partition_view<tile=(8), !cuda_tile.tensor_view<128xf32, strides=[1]>>, %token: !cuda_tile.token) {
    %0 = constant <i32: 0> : !cuda_tile.tile<i32>
    // CHECK: %{{.+}} = store_view_tko weak %{{.+}}, %{{.+}}[%{{.+}}] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    %1 = store_view_tko weak %arg0, %arg1[%0] : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
  
    // CHECK-NEXT: %{{.+}} = store_view_tko weak %{{.+}}, %{{.+}}[%{{.+}}] token = %{{.+}} optimization_hints = <sm_100 = {allow_tma = false, latency = 5}> : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    %2 = store_view_tko weak %arg0, %arg1[%0] token = %token optimization_hints=<sm_100={allow_tma = false, latency = 5}> : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
  
    // CHECK-NEXT: %{{.+}} = store_view_tko relaxed device %{{.+}}, %{{.+}}[%{{.+}}] token = %{{.+}} optimization_hints = <sm_100 = {allow_tma = false}, sm_120 = {latency = 5}> : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    %3 = store_view_tko relaxed device %arg0, %arg1[%0] token = %token optimization_hints=<sm_100={allow_tma = false}, sm_120={latency = 5}> : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
    return
  }
  cuda_tile.experimental$func @load(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %t1 : !cuda_tile.token) -> !cuda_tile.token {
    // CHECK: %result, %result_token = load_ptr_tko weak %arg0 token=%arg1 optimization_hints = <sm_100 = {latency = 5}> : tile<16x32xptr<f32>> -> tile<16x32xf32>, token
    %0, %t2 = cuda_tile.load_ptr_tko weak %arg0 token=%t1 optimization_hints = <sm_100={latency = 5}>
      : !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>> -> !cuda_tile.tile<16x32xf32>, !cuda_tile.token
    return %t2 : !cuda_tile.token
  }
  cuda_tile.experimental$func @store_ptr_tko(%arg0: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %arg1 : !cuda_tile.tile<16x32xf32>, %t1 : !cuda_tile.token) -> !cuda_tile.token {
    // CHECK: %0 = store_ptr_tko weak %arg0, %arg1 token=%arg2 optimization_hints = <sm_100 = {latency = 5}> : tile<16x32xptr<f32>>, tile<16x32xf32> -> token
    %t2 = cuda_tile.store_ptr_tko weak %arg0, %arg1 token=%t1 optimization_hints=<sm_100={latency = 5}>
      : !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, !cuda_tile.tile<16x32xf32> -> !cuda_tile.token
    return %t2 : !cuda_tile.token
  }
}
