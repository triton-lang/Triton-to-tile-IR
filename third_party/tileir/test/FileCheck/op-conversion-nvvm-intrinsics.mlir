// RUN: triton-cuda-tile-opt %s -split-input-file -verify-diagnostics --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=true flush-to-zero-modifier=true num-warps-in-cta=8},reconcile-unrealized-casts)" | FileCheck %s
// RUN: triton-cuda-tile-opt %s -split-input-file -verify-diagnostics --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=false flush-to-zero-modifier=false num-warps-in-cta=4},reconcile-unrealized-casts)" | FileCheck %s

// NVVM intrinsic symbols emitted by triton.language.extra.cuda lower to the
// same native ops as the pure inline-asm forms, independent of the general
// approximation/FTZ pass options.

module {
  tt.func public @nvvm_exp2_ftz(%x: f32, %out: !tt.ptr<f32>) {
    %r = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.ex2.approx.ftz.f32"} : (f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_exp2_ftz
// CHECK: exp2 %{{[^ ]+}} flush_to_zero : tile<f32>
// CHECK-NOT: extern_elementwise

// -----
module {
  tt.func public @nvvm_tf32_rn(%x: f32, %out: !tt.ptr<f32>) {
    %r = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.f2tf32.rn"} : (f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_tf32_rn
// CHECK-NOT: nearest_away
// CHECK: %[[TF32:.*]] = ftof %{{[^ ]+}} : tile<f32> -> tile<tf32>
// CHECK: ftof %[[TF32]] : tile<tf32> -> tile<f32>
// CHECK-NOT: extern_elementwise

// -----
module {
  tt.func public @nvvm_tf32_rna(%x: f32, %out: !tt.ptr<f32>) {
    %r = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.f2tf32.rna"} : (f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_tf32_rna
// CHECK: %[[TF32:.*]] = ftof %{{[^ ]+}} rounding<nearest_away> : tile<f32> -> tile<tf32>
// CHECK: ftof %[[TF32]] : tile<tf32> -> tile<f32>
// CHECK-NOT: extern_elementwise

// -----
module {
  tt.func public @nvvm_fmax_nan_xorsign_abs(%a: f32, %b: f32, %out: !tt.ptr<f32>) {
    %r = tt.extern_elementwise %a, %b {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.fmax.nan.xorsign.abs.f"} : (f32, f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_fmax_nan_xorsign_abs
// CHECK: xori
// CHECK: %[[MAG:.*]] = maxf {{.*}} propagate_nan : tile<f32>
// CHECK: %[[MB:.*]] = bitcast %[[MAG]] : tile<f32> -> tile<i32>
// CHECK: cmpi {{.*}} unsigned
// CHECK: %[[SIGNED:.*]] = ori %[[MB]],
// CHECK: select {{.*}}%[[MB]], %[[SIGNED]]
// CHECK-NOT: flush_to_zero
// CHECK-NOT: extern_elementwise

// -----
module {
  tt.func public @nvvm_fmax_xorsign_abs(%a: f32, %b: f32, %out: !tt.ptr<f32>) {
    %r = tt.extern_elementwise %a, %b {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.fmax.xorsign.abs.f"} : (f32, f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_fmax_xorsign_abs
// CHECK-NOT: propagate_nan
// CHECK: xori
// CHECK: %[[MAG:.*]] = maxf %{{[^ ]+}}, %{{[^ ]+}} : tile<f32>
// CHECK: %[[MB:.*]] = bitcast %[[MAG]] : tile<f32> -> tile<i32>
// CHECK: cmpi {{.*}} unsigned
// CHECK: %[[SIGNED:.*]] = ori %[[MB]],
// CHECK: select {{.*}}%[[MB]], %[[SIGNED]]
// CHECK-NOT: propagate_nan
// CHECK-NOT: extern_elementwise

// -----
module {
  tt.func public @nvvm_fmin_nan_xorsign_abs(%a: f32, %b: f32, %out: !tt.ptr<f32>) {
    %r = tt.extern_elementwise %a, %b {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.fmin.nan.xorsign.abs.f"} : (f32, f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_fmin_nan_xorsign_abs
// CHECK: xori
// CHECK: %[[MAG:.*]] = minf {{.*}} propagate_nan : tile<f32>
// CHECK: %[[MB:.*]] = bitcast %[[MAG]] : tile<f32> -> tile<i32>
// CHECK: %[[SIGNED:.*]] = ori %[[MB]],
// CHECK: select {{.*}}%[[MB]], %[[SIGNED]]
// CHECK-NOT: extern_elementwise

// -----
module {
  tt.func public @nvvm_fmin_xorsign_abs(%a: f32, %b: f32, %out: !tt.ptr<f32>) {
    %r = tt.extern_elementwise %a, %b {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.fmin.xorsign.abs.f"} : (f32, f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_fmin_xorsign_abs
// CHECK-NOT: propagate_nan
// CHECK: %[[MAG:.*]] = minf %{{[^ ]+}}, %{{[^ ]+}} : tile<f32>
// CHECK: %[[MB:.*]] = bitcast %[[MAG]] : tile<f32> -> tile<i32>
// CHECK: %[[SIGNED:.*]] = ori %[[MB]],
// CHECK: select {{.*}}%[[MB]], %[[SIGNED]]
// CHECK-NOT: propagate_nan
// CHECK-NOT: extern_elementwise

// -----
// The intrinsic returns the packed byte zero-extended to i16. Tensor packing
// also checks the operand-to-nibble ordering across rows.
module {
  tt.func public @nvvm_fp4_pack(%hi: f32, %lo: f32, %out: !tt.ptr<i16>) {
    %h = tt.splat %hi : f32 -> tensor<2x4xf32>
    %l = tt.splat %lo : f32 -> tensor<2x4xf32>
    %false = arith.constant false
    %f = tt.splat %false : i1 -> tensor<2x4xi1>
    %r = tt.extern_elementwise %h, %l, %f {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.ff.to.e2m1x2.rn.satfinite"} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xi1>) -> tensor<2x4xi16>
    %p = tt.splat %out : !tt.ptr<i16> -> tensor<2x4x!tt.ptr<i16>>
    tt.store %p, %r : tensor<2x4x!tt.ptr<i16>>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_fp4_pack
// CHECK: %[[HI:.*]] = broadcast {{.*}} -> tile<2x4xf32>
// CHECK: %[[LO:.*]] = broadcast {{.*}} -> tile<2x4xf32>
// CHECK: %[[L:.*]] = reshape %[[LO]] : tile<2x4xf32> -> tile<8x1xf32>
// CHECK: %[[H:.*]] = reshape %[[HI]] : tile<2x4xf32> -> tile<8x1xf32>
// CHECK: %[[PAIR:.*]] = cat %[[L]], %[[H]] dim = 1
// CHECK: %[[FLAT:.*]] = reshape %[[PAIR]] : tile<8x2xf32> -> tile<16xf32>
// CHECK: %[[F4:.*]] = ftof %[[FLAT]] : tile<16xf32> -> tile<16xf4E2M1FN>
// CHECK: %[[BYTES:.*]] = pack %[[F4]] : tile<16xf4E2M1FN> -> tile<8xi8>
// CHECK: %[[SHAPED:.*]] = reshape %[[BYTES]] : tile<8xi8> -> tile<2x4xi8>
// CHECK: exti %[[SHAPED]] unsigned : tile<2x4xi8> -> tile<2x4xi16>
// CHECK-NOT: extern_elementwise

// -----
module {
  tt.func public @nvvm_fp4_pack_dense_false(%hi: f32, %lo: f32, %out: !tt.ptr<i16>) {
    %h = tt.splat %hi : f32 -> tensor<2x4xf32>
    %l = tt.splat %lo : f32 -> tensor<2x4xf32>
    %f = arith.constant dense<false> : tensor<2x4xi1>
    %r = tt.extern_elementwise %h, %l, %f {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.ff.to.e2m1x2.rn.satfinite"} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xi1>) -> tensor<2x4xi16>
    %p = tt.splat %out : !tt.ptr<i16> -> tensor<2x4x!tt.ptr<i16>>
    tt.store %p, %r : tensor<2x4x!tt.ptr<i16>>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_fp4_pack_dense_false
// CHECK: %[[BYTES:.*]] = pack %{{[^ ]+}} : tile<16xf4E2M1FN> -> tile<8xi8>
// CHECK: %[[SHAPED:.*]] = reshape %[[BYTES]] : tile<8xi8> -> tile<2x4xi8>
// CHECK: exti %[[SHAPED]] unsigned : tile<2x4xi8> -> tile<2x4xi16>
// CHECK-NOT: extern_elementwise

// -----
module {
  tt.func public @nvvm_fp4_pack_scalar(%hi: f32, %lo: f32, %out: !tt.ptr<i16>) {
    %false = arith.constant false
    %r = tt.extern_elementwise %hi, %lo, %false {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.ff.to.e2m1x2.rn.satfinite"} : (f32, f32, i1) -> i16
    tt.store %out, %r : !tt.ptr<i16>
    tt.return
  }
}
// CHECK-LABEL: entry @nvvm_fp4_pack_scalar
// CHECK: %[[BYTES:.*]] = pack %{{[^ ]+}} : tile<2xf4E2M1FN> -> tile<1xi8>
// CHECK: %[[SHAPED:.*]] = reshape %[[BYTES]] : tile<1xi8> -> tile<i8>
// CHECK: exti %[[SHAPED]] unsigned : tile<i8> -> tile<i16>
// CHECK-NOT: extern_elementwise

// -----
// Only the plain form of the pack instruction has a native equivalent.
module {
  tt.func public @reject_fp4_pack_flag_true(%hi: f32, %lo: f32, %out: !tt.ptr<i16>) {
    %true = arith.constant true
    // expected-error@+1 {{failed to legalize operation 'tt.extern_elementwise'}}
    %r = tt.extern_elementwise %hi, %lo, %true {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.ff.to.e2m1x2.rn.satfinite"} : (f32, f32, i1) -> i16
    tt.store %out, %r : !tt.ptr<i16>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_fp4_pack_flag_dynamic(%hi: f32, %lo: f32, %flag: i1, %out: !tt.ptr<i16>) {
    // expected-error@+1 {{failed to legalize operation 'tt.extern_elementwise'}}
    %r = tt.extern_elementwise %hi, %lo, %flag {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.ff.to.e2m1x2.rn.satfinite"} : (f32, f32, i1) -> i16
    tt.store %out, %r : !tt.ptr<i16>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_fp4_pack_narrow_result(%hi: f32, %lo: f32, %out: !tt.ptr<i8>) {
    %false = arith.constant false
    // expected-error@+1 {{failed to legalize operation 'tt.extern_elementwise'}}
    %r = tt.extern_elementwise %hi, %lo, %false {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.ff.to.e2m1x2.rn.satfinite"} : (f32, f32, i1) -> i8
    tt.store %out, %r : !tt.ptr<i8>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_tf32_other_rounding(%x: f32, %out: !tt.ptr<f32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.extern_elementwise'}}
    %r = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.f2tf32.rz"} : (f32) -> f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}

// -----
module {
  tt.func public @reject_exp2_wrong_type(%x: f16, %out: !tt.ptr<f16>) {
    // expected-error@+1 {{failed to legalize operation 'tt.extern_elementwise'}}
    %r = tt.extern_elementwise %x {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.ex2.approx.ftz.f32"} : (f16) -> f16
    tt.store %out, %r : !tt.ptr<f16>
    tt.return
  }
}

// -----
// Special-register reads have no public cuda_tile equivalent.
module {
  tt.func public @reject_smid(%out: !tt.ptr<i32>) {
    // expected-error@+1 {{failed to legalize operation 'tt.extern_elementwise'}}
    %r = tt.extern_elementwise {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.read.ptx.sreg.smid"} : () -> i32
    tt.store %out, %r : !tt.ptr<i32>
    tt.return
  }
}
