// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// expected-error @below{{expects 2 arguments in the body, but got 1}}
%0 = cuda_tile.experimental$generate {
^bb0(%arg0: !cuda_tile.tile<i32>):
  cuda_tile.yield %arg0 : !cuda_tile.tile<i32>
} : !cuda_tile.tile<16x16xi32>

// -----

// expected-error @below{{expects all arg types to be 0-rank tile of i32}}
%0 = cuda_tile.experimental$generate {
^bb0(%arg0: !cuda_tile.tile<i32>, %arg1: !cuda_tile.tile<i64>):
  cuda_tile.yield %arg0 : !cuda_tile.tile<i32>
} : !cuda_tile.tile<16x16xi32>

// -----

// expected-error @below{{expects exactly 1 yielded value from the region}}
%0 = cuda_tile.experimental$generate {
^bb0(%arg0: !cuda_tile.tile<i32>, %arg1: !cuda_tile.tile<i32>):
  cuda_tile.yield %arg0, %arg1 : !cuda_tile.tile<i32>, !cuda_tile.tile<i32>
} : !cuda_tile.tile<16x16xi32>

// -----

// expected-error @below{{invalid yielded value from the region}}
%0 = cuda_tile.experimental$generate {
^bb0(%arg0: !cuda_tile.tile<i32>, %arg1: !cuda_tile.tile<i32>):
  %1 = cuda_tile.constant <i64: 0> : !cuda_tile.tile<i64>
  cuda_tile.yield %1 : !cuda_tile.tile<i64>
} : !cuda_tile.tile<16x16xi32>

// -----

// expected-error @below{{expects 0d tile in the body}}
%0 = cuda_tile.experimental$generate {
^bb0(%arg0: !cuda_tile.tile<i32>, %arg1: !cuda_tile.tile<i32>):
  %1 = cuda_tile.addi %arg0, %arg1 : !cuda_tile.tile<i32>
  // expected-note @below{{op with higher dimensionality found}}
  %2 = cuda_tile.constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
  cuda_tile.yield %1 : !cuda_tile.tile<i32>
} : !cuda_tile.tile<16x16xi32>

// -----

// expected-error @+1 {{expected either IR string or region}}
cuda_tile.experimental$inject_ir {stage = "llvm", ir = "gpu.printf \"World\""} {
  cuda_tile.print_tko "World" -> !cuda_tile.token
}

// -----

// expected-error @+1 {{unsupported stage, expected 'nv_tileaa', 'nv_tileas' or 'llvm'}}
cuda_tile.experimental$inject_ir {stage = "foo", ir = "gpu.printf \"World\""} {}

// -----

%0 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{'cuda_tile.experimental$log10' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<i32>'}}
cuda_tile.experimental$log10 %0 : !cuda_tile.tile<i32>

// -----

%0 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
// expected-error @below{{'cuda_tile.experimental$log1p' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<i32>'}}
cuda_tile.experimental$log1p %0 : !cuda_tile.tile<i32>

// -----

cuda_tile.entry @test_mark_for_reuse_single_queue() {
    %queue0 = cuda_tile.experimental$create_queue : <[!cuda_tile.tile<128xf32>]>
    // expected-error @below{{expect at least 2 queues, but got: 1}}
    cuda_tile.experimental$mark_for_reuse %queue0 <{partitions = array<i32: 1>}> : !cuda_tile.queue<[!cuda_tile.tile<128xf32>]>
}

// -----

cuda_tile.module @kernels {
    experimental$func @test_mark_for_reuse_empty_partitions(%queue0: !cuda_tile.queue<[!cuda_tile.tile<128xf32>]>, %queue1: !cuda_tile.queue<[!cuda_tile.tile<128xf32>]>) {
        // expected-error @below{{expect sum(partitions) == 2, but got: 1}}
        cuda_tile.experimental$mark_for_reuse %queue0, %queue1 <{partitions = array<i32: 1>}>: !cuda_tile.queue<[!cuda_tile.tile<128xf32>]>, !cuda_tile.queue<[!cuda_tile.tile<128xf32>]>
        return
    }
}

// -----

cuda_tile.module @kernels {
    cuda_tile.entry @test_mark_for_reuse_different_depths() {
        %queue0 = cuda_tile.experimental$create_queue : <[!cuda_tile.tile<128xf32>], depth=1>
        %queue1 = cuda_tile.experimental$create_queue : <[!cuda_tile.tile<128xf32>], depth=2>
        // expected-error @below{{expect all operands have the same depth, but got 1 vs 2}}
        cuda_tile.experimental$mark_for_reuse %queue0, %queue1 <{partitions = array<i32: 1, 1>}> : !cuda_tile.queue<[!cuda_tile.tile<128xf32>], depth=1>, !cuda_tile.queue<[!cuda_tile.tile<128xf32>], depth=2>
    }
}

// -----

cuda_tile.module @kernels {
  experimental$func @test_tiled_atomic_rmw(%arg0: !cuda_tile.partition_view<tile=(2x16), !cuda_tile.tensor_view<16x16xi32, strides=[16, 1]>>, %arg1: !cuda_tile.tile<2x16xi32>) {
    %idx0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %idx1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // expected-error @below{{'cuda_tile.experimental$tiled_atomic_rmw_unordered' op unsupported atomic RMW mode: xchg}}
    %result_0, %result_token_0 = experimental$tiled_atomic_rmw_unordered weak %arg0[%idx0, %idx1],xchg, %arg1 attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} : !cuda_tile.partition_view<tile=(2x16), !cuda_tile.tensor_view<16x16xi32, strides=[16, 1]>>, tile<i32>,!cuda_tile.tile<2x16xi32> -> !cuda_tile.tile<2x16xi32>, token
  }
}

// -----

cuda_tile.module @test_static_alloca {
  cuda_tile.entry @test_static_alloca(%c64: tile<i64>) {
    // expected-error @below {{failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0}}
    %0 = experimental$alloca num_elem = -10, alignment = 16 : tile<ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  cuda_tile.entry @test_mixed_alloca(%c64: tile<i64>) {
    // expected-error @below {{failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0}}
    %0 = experimental$alloca num_elem = 32, alignment = -10 : tile<ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  cuda_tile.entry @test_alloc_align() {
    // expected-error @below {{op 'alignment' must be power of two}}
    %0 = experimental$alloca num_elem = 64, alignment = 3 : tile<ptr<f32>>
  }
}

// -----

cuda_tile.module @kernels {
  cuda_tile.entry @test_alloc_align() {
    // expected-error @below {{'alignment' (2) must be at least the natural size (4 bytes) for element type 'f32'}}
    %0 = experimental$alloca num_elem = 64, alignment = 2 : tile<ptr<f32>>
  }
}

// -----

cuda_tile.module @test_mmaf_scaled {
  experimental$func @test_mmaf_scaled_fp8e4m3_scale(
    %arg0: !cuda_tile.tile<128x128xf8E5M2>,
    %arg1: !cuda_tile.tile<128x128xf8E5M2>,
    %arg2: !cuda_tile.tile<128x128xf32>,
    %arg3: !cuda_tile.tile<128x4xf8E4M3FN>,
    %arg4: !cuda_tile.tile<4x128xf8E4M3FN>
  ) {
    // expected-error @below {{op unsupported combination of element and scale factor types}}
    %0 = experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
      !cuda_tile.tile<128x128xf8E5M2>,
      !cuda_tile.tile<128x128xf8E5M2>,
      !cuda_tile.tile<128x128xf32>,
      !cuda_tile.tile<128x4xf8E4M3FN>,
      !cuda_tile.tile<4x128xf8E4M3FN>
      -> !cuda_tile.tile<128x128xf32>
  }

  experimental$func @test_mmaf_scaled_fp16_acc(
        %arg0: !cuda_tile.tile<128x128xf8E5M2>,
        %arg1: !cuda_tile.tile<128x128xf8E5M2>,
        %arg2: !cuda_tile.tile<128x128xf16>,
        %arg3: !cuda_tile.tile<128x4xf8E8M0FNU>,
        %arg4: !cuda_tile.tile<4x128xf8E8M0FNU>
  ) {
    // expected-error @below {{op operand #2 must be tile of f32 values}}
    %0 = experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
        !cuda_tile.tile<128x128xf8E5M2>,
        !cuda_tile.tile<128x128xf8E5M2>,
        !cuda_tile.tile<128x128xf16>,
        !cuda_tile.tile<128x4xf8E8M0FNU>,
        !cuda_tile.tile<4x128xf8E8M0FNU>
        -> !cuda_tile.tile<128x128xf32>
  }

  experimental$func @test_mmaf_scaled_fp16_res(
        %arg0: !cuda_tile.tile<128x128xf8E5M2>,
        %arg1: !cuda_tile.tile<128x128xf8E5M2>,
        %arg2: !cuda_tile.tile<128x128xf32>,
        %arg3: !cuda_tile.tile<128x4xf8E8M0FNU>,
        %arg4: !cuda_tile.tile<4x128xf8E8M0FNU>
  ) {
    // expected-error @below {{op result #0 must be tile of f32 values}}
    %0 = experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
        !cuda_tile.tile<128x128xf8E5M2>,
        !cuda_tile.tile<128x128xf8E5M2>,
        !cuda_tile.tile<128x128xf32>,
        !cuda_tile.tile<128x4xf8E8M0FNU>,
        !cuda_tile.tile<4x128xf8E8M0FNU>
        -> !cuda_tile.tile<128x128xf16>
  }

  experimental$func @test_mmaf_scaled_mixed_input_types(
        %arg0: !cuda_tile.tile<128x128xf4E2M1FN>,
        %arg1: !cuda_tile.tile<128x128xf8E5M2>,
        %arg2: !cuda_tile.tile<128x128xf32>,
        %arg3: !cuda_tile.tile<128x8xf8E8M0FNU>,
        %arg4: !cuda_tile.tile<8x128xf8E8M0FNU>
  ) {
    // expected-error @below {{op failed to verify that all of {a, b} have the same element type}}
    %0 = experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
        !cuda_tile.tile<128x128xf4E2M1FN>,
        !cuda_tile.tile<128x128xf8E5M2>,
        !cuda_tile.tile<128x128xf32>,
        !cuda_tile.tile<128x8xf8E8M0FNU>,
        !cuda_tile.tile<8x128xf8E8M0FNU>
        -> !cuda_tile.tile<128x128xf32>
  }

  experimental$func @test_mmaf_scaled_mixed_scale_types(
        %arg0: !cuda_tile.tile<128x128xf4E2M1FN>,
        %arg1: !cuda_tile.tile<128x128xf4E2M1FN>,
        %arg2: !cuda_tile.tile<128x128xf32>,
        %arg3: !cuda_tile.tile<128x8xf8E8M0FNU>,
        %arg4: !cuda_tile.tile<8x128xf8E4M3FN>
  ) {
    // expected-error @below {{op failed to verify that all of {sfa, sfb} have the same element type}}
    %0 = experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
        !cuda_tile.tile<128x128xf4E2M1FN>,
        !cuda_tile.tile<128x128xf4E2M1FN>,
        !cuda_tile.tile<128x128xf32>,
        !cuda_tile.tile<128x8xf8E8M0FNU>,
        !cuda_tile.tile<8x128xf8E4M3FN>
        -> !cuda_tile.tile<128x128xf32>
  }
}

// -----

cuda_tile.module @kernels {
  experimental$func @test_reinterpret_bitcast_dtype(%arg0: !cuda_tile.tile<128x64xi8>) {
    // expected-error @below {{op source element type must be wider than result element type}}
    %0 = experimental$reinterpret %arg0 : !cuda_tile.tile<128x64xi8> -> !cuda_tile.tile<128x64xf8E5M2>
  }

  experimental$func @test_reinterpret_narrow_to_wide_dtype(%arg0: !cuda_tile.tile<128x64xi8>) {
    // expected-error @below {{op source element type must be wider than result element type}}
    %0 = experimental$reinterpret %arg0 : !cuda_tile.tile<128x64xi8> -> !cuda_tile.tile<64x64xi16>
  }

  experimental$func @test_reinterpret_change_multi_dim(%arg0: !cuda_tile.tile<128x64xi8>) {
    // expected-error @below {{op types and shapes mismatch}}
    %0 = experimental$reinterpret %arg0 : !cuda_tile.tile<128x64xi8> -> !cuda_tile.tile<512x32xf4E2M1FN>
  }

  experimental$func @test_reinterpret_additional_dim(%arg0: !cuda_tile.tile<128x64xi8>) {
    // expected-error @below {{op failed to verify that all of {source, result} have same rank}}
    %0 = experimental$reinterpret %arg0 : !cuda_tile.tile<128x64xi8> -> !cuda_tile.tile<128x64x2xi16>
  }

  experimental$func @test_reinterpret_byte_mismatch(%arg0: !cuda_tile.tile<128x64xi8>) {
    // expected-error @below {{op types and shapes mismatch}}
    %0 = experimental$reinterpret %arg0 : !cuda_tile.tile<128x64xi8> -> !cuda_tile.tile<128x256xf4E2M1FN>
  }
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32xf16>) {
  // expected-error @below {{op failed to verify that all of {source, result} have same rank}}
  %0 = experimental$pack %arg0 : tile<32xf16> -> tile<32x2xi8>
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32x32xf16>) {
  // expected-error @below {{op expects source and result to be rank-1 tiles}}
  %0 = experimental$pack %arg0 : tile<32x32xf16> -> tile<32x64xi8>
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32xf16>) {
  // expected-error @below {{op expects source and result to have the same size in bytes, but got source tile size 64 bytes and result tile size 32 bytes}}
  %0 = experimental$pack %arg0 : tile<32xf16> -> tile<32xi8>
}

// -----

cuda_tile.testing$func @test_pack_op(%arg0: !cuda_tile.tile<32xf4E2M1FN>) {
  // expected-error @below {{op expects the source element type to be wider than the result element type}}
  %0 = experimental$pack %arg0 : tile<32xf4E2M1FN> -> tile<16xi8>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<64xi8>) {
  // expected-error @below {{op failed to verify that all of {source, result} have same rank}}
  %0 = experimental$unpack %arg0 : tile<64xi8> -> tile<32x2xf16>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<64x64xi8>) {
  // expected-error @below {{op expects source and result to be rank-1 tiles}}
  %0 = experimental$unpack %arg0 : tile<64x64xi8> -> tile<64x32xf16>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<64xi8>) {
  // expected-error @below {{op expects source and result to have the same size in bytes, but got source tile size 64 bytes and result tile size 128 bytes}}
  %0 = experimental$unpack %arg0 : tile<64xi8> -> tile<64xf16>
}

// -----

cuda_tile.testing$func @test_unpack_op(%arg0: !cuda_tile.tile<64xi8>) {
  // expected-error @below {{op expects the source element type to be wider than the result element type}}
  %0 = experimental$unpack %arg0 : tile<64xi8> -> tile<32xf16>
}
