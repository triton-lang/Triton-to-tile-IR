// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  cuda_tile.experimental$func @valid_array_tensor_func(%arg0: !cuda_tile.tile<2xi32>) -> !cuda_tile.tile<i32> {
    %0 = cuda_tile.constant <i32: 5> : !cuda_tile.tile<i32>
    cuda_tile.return %0 : !cuda_tile.tile<i32>
  }
}
