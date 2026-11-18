// RUN: triton-cuda-tile-opt %s -split-input-file -verify-diagnostics --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=true flush-to-zero-modifier=true num-warps-in-cta=4},reconcile-unrealized-casts)" | FileCheck %s

// PTX div.full.f32 written as inline asm lowers to a native full-range
// division. The instruction names its own rounding and has no .ftz suffix, so
// neither pass modifier applies to it.
module {
  tt.func public @div_full(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base_a = tt.splat %a : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptrs_a = tt.addptr %base_a, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %x = tt.load %ptrs_a : tensor<8x!tt.ptr<f32>>
    %base_b = tt.splat %b : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptrs_b = tt.addptr %base_b, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %y = tt.load %ptrs_b : tensor<8x!tt.ptr<f32>>
    %q = tt.elementwise_inline_asm "div.full.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %x, %y : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %base_o = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptrs_o = tt.addptr %base_o, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %ptrs_o, %q : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: entry @div_full
// CHECK-NOT: elementwise_inline_asm
// CHECK: divf {{.*}}rounding<full> : tile<8xf32>
// CHECK-NOT: elementwise_inline_asm

// -----
// Other division instructions stay illegal: only the exact form is matched.
module {
  tt.func public @reject_div_approx(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base_a = tt.splat %a : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptrs_a = tt.addptr %base_a, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %x = tt.load %ptrs_a : tensor<8x!tt.ptr<f32>>
    %base_b = tt.splat %b : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptrs_b = tt.addptr %base_b, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %y = tt.load %ptrs_b : tensor<8x!tt.ptr<f32>>
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %q = tt.elementwise_inline_asm "div.rn.f32 $0, $1, $2;" {constraints = "=f,f,f", packed_element = 1 : i32, pure = true} %x, %y : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    %base_o = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptrs_o = tt.addptr %base_o, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %ptrs_o, %q : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
