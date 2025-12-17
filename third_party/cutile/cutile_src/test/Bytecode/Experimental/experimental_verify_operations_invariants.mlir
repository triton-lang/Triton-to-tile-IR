// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -split-input-file -verify-diagnostics -allow-unregistered-dialect %s

cuda_tile.module @kernels {
  experimental$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{only pure operations allowed}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=0 reverse=false identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xf32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<f32>, %arg1_prev_iter_arg : !cuda_tile.tile<f32>) {
      // expected-remark @below{{invalid op}}
      cuda_tile.print_tko "hello_world" -> !cuda_tile.token
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<f32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  experimental$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{only pure operations allowed}}
    %0:2 = cuda_tile.reduce %arg0, %arg1
    dim=0 identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>, !cuda_tile.tile<f32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<f32>, %arg1_prev_iter_arg : !cuda_tile.tile<f32>) {
      // expected-remark @below{{invalid op}}
      cuda_tile.print_tko "hello_world" -> !cuda_tile.token
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<f32>
    }
  }
}
