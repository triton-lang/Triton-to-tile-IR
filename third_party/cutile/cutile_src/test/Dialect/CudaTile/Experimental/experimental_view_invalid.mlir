// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// ****************** cuda_tile.experimental$make_strided_view ******************
// expected-error @below{{expected dim_map to map exactly all 2 dimensions of the tile, got 1 mappings}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[0]>

// -----

// expected-error @below{{target dimension is outside of tensor view dimensions, expected strictly less than 2, got 2}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[2, 1]>

// -----

// expected-error @below{{target dimension 0 mapped at least twice (for tile dimensions 0 and 1)}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[0, 0]>

// -----

// expected-error @below{{tile shape dimensions must have power of two length but got [5, 1024]}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(5x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>

// -----

// expected-error @below{{tile dimension 0 exceeds i32 limitations (got 1099511627776, expected strictly positive and less than or equal to 2147483647)}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(1099511627776x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>

// -----

// expected-error @below{{expected tensor_view rank and tile rank to match, got tensor_view of rank 3 and tiles of rank 2}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(1x1), traversal_strides=[1024,1], !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>

// -----

// expected-error @below{{0-dimension tile shape is not supported}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(), traversal_strides=[], !cuda_tile.tensor_view<f32>>

// -----

// expected-error @below{{target dimension must not be negative, got -1}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>, dim_map=[-1, 1]>

// -----

// expected-error @below{{expected 2 traversal strides, got 1}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>

// -----

// expected-error @below{{traversal strides must be strictly positive but got [1, 0]}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1,0], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>

// -----

// expected-error @below{{traversal strides must be strictly positive but got [1, -1]}}
"use_type"() : () -> !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1,-1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_strided_view_wrong_tensor_view_elem(%tensor_view: !cuda_tile.tensor_view<4096x4096xf64, strides=[4096,1]>) {
    // expected-note @above{{prior use here}}
    // expected-error @below{{expects different type than prior uses}}
    cuda_tile.experimental$make_strided_view %tensor_view : !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @make_strided_view_wrong_tensor_view_shape(%tensor_view: !cuda_tile.tensor_view<4096x2048xf32, strides=[4096,1]>) {
    // expected-note @above{{prior use here}}
    // expected-error @below{{expects different type than prior uses}}
    cuda_tile.experimental$make_strided_view %tensor_view : !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>
  }
}

// -----

// ****************** cuda_tile.load_view_tko ******************
cuda_tile.module @module {
  cuda_tile.testing$func @tile_strided_wrong_load_type(%view: !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected tile type to be '!cuda_tile.tile<1024x1024xf32>' (based on view type), got '!cuda_tile.tile<8xf32>'}}
    load_view_tko weak %view[%c0, %c0] : strided_view<tile=(1024x1024), traversal_strides=[1024,1], tensor_view<4096x4096xf32, strides=[4096,1]>>, tile<i32> -> tile<8xf32>, token
  }
}

// -----

// This test uses generic format to test the verifier itself, as the parser already requires this property.
cuda_tile.module @module {
  cuda_tile.testing$func @tile_strided_wrong_load_rank(%view: !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected 2 index operands (based on view type), got 1}}
    "cuda_tile.load_view_tko"(%view, %c0) <{memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], tensor_view<4096x4096xf32, strides=[4096,1]>>, !cuda_tile.tile<i32>) -> (!cuda_tile.tile<1024x1024xf32>, !cuda_tile.token)
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @load_view_tko_index_type_mismatch(%view: !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>) {
    %c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c0_i64 = cuda_tile.constant <i64: 0> : !cuda_tile.tile<i64>
    // expected-error @below{{expected index type 1 to be the same as other index types ('!cuda_tile.tile<i32>'), got '!cuda_tile.tile<i64>'}}
    %x, %t = "cuda_tile.load_view_tko"(%view, %c0_i32, %c0_i64) <{memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 2, 0>}> : (!cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], tensor_view<4096x4096xf32, strides=[4096,1]>>, !cuda_tile.tile<i32>, !cuda_tile.tile<i64>) -> (!cuda_tile.tile<1024x1024xf32>, !cuda_tile.token)
  }
}

// -----

cuda_tile.module @kernels {
  cuda_tile.testing$func @load_missing_index(%memref_i8: !cuda_tile.tensor_view<1024xi8, strides=[1]>) {
    %view_i8 = experimental$make_strided_view %memref_i8 : strided_view<tile=(128), traversal_strides=[1], tensor_view<1024xi8, strides=[1]>>
    // expected-error @below{{expected 1 index operands (based on view type), got 0}}
    %tile_i8_l, %tok_i8 = load_view_tko weak %view_i8[] : strided_view<tile=(128), traversal_strides=[1], tensor_view<1024xi8, strides=[1]>>, tile<i32> -> tile<128xi8>, token
  }
}

// -----

// ****************** cuda_tile.store_view_tko ******************

// This test uses generic format to test the verifier itself, as the parser already requires this property.
cuda_tile.module @module {
  cuda_tile.testing$func @tile_strided_wrong_store_rank(%view: !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>, %tile: !cuda_tile.tile<1024x1024xf32>) {
    %c0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    // expected-error @below{{expected 2 index operands (based on view type), got 1}}
    "cuda_tile.store_view_tko"(%tile, %view, %c0) <{memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> : (!cuda_tile.tile<1024x1024xf32>, !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], tensor_view<4096x4096xf32, strides=[4096,1]>>, !cuda_tile.tile<i32>) -> !cuda_tile.token
  }
}

// -----

cuda_tile.module @module {
  cuda_tile.testing$func @store_view_tko_index_type_mismatch(%view: !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], !cuda_tile.tensor_view<4096x4096xf32, strides=[4096,1]>>, %tile: !cuda_tile.tile<1024x1024xf32>) {
    %c0_i32 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %c0_i64 = cuda_tile.constant <i64: 0> : !cuda_tile.tile<i64>
    // expected-error @below{{expected index type 1 to be the same as other index types ('!cuda_tile.tile<i32>'), got '!cuda_tile.tile<i64>'}}
    %t = "cuda_tile.store_view_tko"(%tile, %view, %c0_i32, %c0_i64) <{memory_ordering_semantics = 0 : i32, operandSegmentSizes = array<i32: 1, 1, 2, 0>}> : (!cuda_tile.tile<1024x1024xf32>, !cuda_tile.strided_view<tile=(1024x1024), traversal_strides=[1024,1], tensor_view<4096x4096xf32, strides=[4096,1]>>, !cuda_tile.tile<i32>, !cuda_tile.tile<i64>) -> !cuda_tile.token
  }
}
