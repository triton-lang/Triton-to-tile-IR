// RUN: triton-cuda-tile-opt %s -split-input-file -verify-diagnostics --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=false flush-to-zero-modifier=false num-warps-in-cta=4},reconcile-unrealized-casts)" | FileCheck %s

// rint/nearbyint select between floor(x) and ceil(x). The upper neighbour must
// not be computed as floor(x) + 1, which turns the -0.0 result for inputs in
// (-1, 0) into +0.0.

module {
  tt.func public @rint_f32(%x: f32, %out: !tt.ptr<f32>) {
    %r = tt.extern_elementwise %x {libname = "libdevice", libpath = "", pure = true, symbol = "__nv_rintf"} : (f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @rint_f32
// CHECK-NOT: addf
// CHECK: floor %[[X:[^ ]+]]
// CHECK-NOT: addf
// CHECK: ceil %[[X]]
// CHECK-NOT: addf
// CHECK: select
// CHECK-NOT: extern_elementwise

// -----
module {
  tt.func public @nearbyint_f64(%x: f64, %out: !tt.ptr<f64>) {
    %r = tt.extern_elementwise %x {libname = "libdevice", libpath = "", pure = true, symbol = "__nv_nearbyint"} : (f64) -> f64
    tt.store %out, %r : !tt.ptr<f64>
    tt.return
  }
}
// CHECK-LABEL: entry @nearbyint_f64
// CHECK-NOT: addf
// CHECK: floor %[[X:[^ ]+]]
// CHECK-NOT: addf
// CHECK: ceil %[[X]]
// CHECK-NOT: addf
// CHECK: select
// CHECK-NOT: extern_elementwise
