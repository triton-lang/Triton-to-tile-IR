// RUN: triton-cuda-tile-opt %s -split-input-file -verify-diagnostics --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=false flush-to-zero-modifier=false num-warps-in-cta=4},reconcile-unrealized-casts)" | FileCheck %s

// The packed maximum of FP4 magnitude codes used by triton_kernels lowers to
// native integer ops: the largest 3-bit code among the eight nibbles.
module {
  tt.func public @fp4_max_code(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %pi = tt.splat %in : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
    %x = tt.load %pi : tensor<2x4x!tt.ptr<i32>>
    %r = tt.elementwise_inline_asm "{\0A .reg .b32 a, b;\0A and.b32 a, $1, 0x00070007;\0A shr.u32 b, $1, 4;  and.b32 b, b, 0x00070007; max.u16x2 a, a, b;\0A shr.u32 b, $1, 8;  and.b32 b, b, 0x00070007; max.u16x2 a, a, b;\0A shr.u32 b, $1, 12; and.b32 b, b, 0x00070007; max.u16x2 a, a, b;\0A shr.u32 b, a, 16;\0A and.b32 a, a, 7;\0A max.u32 $0, a, b;\0A }" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %x : tensor<2x4xi32> -> tensor<2x4xi32>
    %po = tt.splat %out : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
    tt.store %po, %r : tensor<2x4x!tt.ptr<i32>>
    tt.return
  }
}
// CHECK-LABEL: entry @fp4_max_code
// CHECK-NOT: elementwise_inline_asm
// CHECK: andi
// CHECK: shri {{.*}} unsigned : tile<2x4xi32>
// CHECK: maxi {{.*}} unsigned : tile<2x4xi32>
// CHECK-NOT: elementwise_inline_asm

// -----
// Any other instruction sequence stays illegal: a different mask is not the
// matched form.
module {
  tt.func public @reject_other_mask(%in: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %pi = tt.splat %in : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
    %x = tt.load %pi : tensor<2x4x!tt.ptr<i32>>
    // expected-error@+1 {{failed to legalize operation 'tt.elementwise_inline_asm'}}
    %r = tt.elementwise_inline_asm "{\0A .reg .b32 a, b;\0A and.b32 a, $1, 0x000f000f;\0A shr.u32 b, $1, 4;  and.b32 b, b, 0x000f000f; max.u16x2 a, a, b;\0A shr.u32 b, $1, 8;  and.b32 b, b, 0x000f000f; max.u16x2 a, a, b;\0A shr.u32 b, $1, 12; and.b32 b, b, 0x000f000f; max.u16x2 a, a, b;\0A shr.u32 b, a, 16;\0A and.b32 a, a, 15;\0A max.u32 $0, a, b;\0A }" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %x : tensor<2x4xi32> -> tensor<2x4xi32>
    %po = tt.splat %out : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
    tt.store %po, %r : tensor<2x4x!tt.ptr<i32>>
    tt.return
  }
}
