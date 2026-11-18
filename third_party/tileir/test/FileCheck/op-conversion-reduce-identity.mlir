// RUN: triton-cuda-tile-opt %s -split-input-file -verify-diagnostics --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=false flush-to-zero-modifier=false num-warps-in-cta=4},reconcile-unrealized-casts)" | FileCheck %s

// The identity of a reduce must be neutral for its combine function, because
// the compiler may fold it into any partial result. minnum/maxnum return the
// other operand when one operand is NaN, so only NaN is neutral for them: an
// infinity would replace the NaN result of an all-NaN reduction. The
// NaN-propagating minimum/maximum keep the infinities.

module {
  tt.func public @reduce_maxnum_f32(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %ptrs = tt.splat %in : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %addr = tt.addptr %ptrs, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %addr : tensor<16x!tt.ptr<f32>>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) : (tensor<16xf32>) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @reduce_maxnum_f32
// CHECK: reduce %{{[^ ]+}} dim=0 identities=[0x7FC00000 : f32]
// CHECK-NEXT: (%[[A:[^ ]+]]: tile<f32>, %[[B:[^ ]+]]: tile<f32>)
// CHECK-NEXT: maxf %[[A]], %[[B]] : tile<f32>

// -----
module {
  tt.func public @reduce_minnum_f64(%in: !tt.ptr<f64>, %out: !tt.ptr<f64>) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %ptrs = tt.splat %in : !tt.ptr<f64> -> tensor<16x!tt.ptr<f64>>
    %addr = tt.addptr %ptrs, %range : tensor<16x!tt.ptr<f64>>, tensor<16xi32>
    %x = tt.load %addr : tensor<16x!tt.ptr<f64>>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f64, %b: f64):
      %m = arith.minnumf %a, %b : f64
      tt.reduce.return %m : f64
    }) : (tensor<16xf64>) -> f64
    tt.store %out, %r : !tt.ptr<f64>
    tt.return
  }
}
// CHECK-LABEL: entry @reduce_minnum_f64
// CHECK: reduce %{{[^ ]+}} dim=0 identities=[0x7FF8000000000000 : f64]
// CHECK-NEXT: (%[[A:[^ ]+]]: tile<f64>, %[[B:[^ ]+]]: tile<f64>)
// CHECK-NEXT: minf %[[A]], %[[B]] : tile<f64>

// -----
module {
  tt.func public @reduce_maximum_f32(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %ptrs = tt.splat %in : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %addr = tt.addptr %ptrs, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %addr : tensor<16x!tt.ptr<f32>>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maximumf %a, %b : f32
      tt.reduce.return %m : f32
    }) : (tensor<16xf32>) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @reduce_maximum_f32
// CHECK: reduce %{{[^ ]+}} dim=0 identities=[0xFF800000 : f32]
// CHECK-NEXT: (%[[A:[^ ]+]]: tile<f32>, %[[B:[^ ]+]]: tile<f32>)
// CHECK-NEXT: maxf %[[A]], %[[B]] propagate_nan : tile<f32>

// -----
module {
  tt.func public @reduce_minimum_f32(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %ptrs = tt.splat %in : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %addr = tt.addptr %ptrs, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %x = tt.load %addr : tensor<16x!tt.ptr<f32>>
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.minimumf %a, %b : f32
      tt.reduce.return %m : f32
    }) : (tensor<16xf32>) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @reduce_minimum_f32
// CHECK: reduce %{{[^ ]+}} dim=0 identities=[0x7F800000 : f32]
// CHECK-NEXT: (%[[A:[^ ]+]]: tile<f32>, %[[B:[^ ]+]]: tile<f32>)
// CHECK-NEXT: minf %[[A]], %[[B]] propagate_nan : tile<f32>
