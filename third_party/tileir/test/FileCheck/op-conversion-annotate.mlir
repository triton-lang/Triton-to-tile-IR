// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,reconcile-unrealized-casts)" | FileCheck %s

// tt.annotate carries a hint for one result of a multi-result producer. The
// hint becomes an assume on that result only; the op itself is an identity.
module {
  tt.func public @annotate_one_result(%p: !tt.ptr<i32>, %c: i1, %x: i32, %y: i32) {
    %r0, %r1 = scf.if %c -> (i32, i32) {
      scf.yield %x, %y : i32, i32
    } else {
      scf.yield %y, %x : i32, i32
    }
    %a = tt.annotate %r0 {tt.divisibility = dense<16> : tensor<1xi32>} : i32
    %s = arith.addi %a, %r1 : i32
    tt.store %p, %s : !tt.ptr<i32>
    tt.return
  }
}
// CHECK-LABEL: entry @annotate_one_result
// CHECK: %[[R:[0-9]+]]:2 = if
// CHECK: %[[A:.*]] = assume div_by<16>, %[[R]]#0 : tile<i32>
// CHECK-NOT: assume
// CHECK: addi %[[A]], %[[R]]#1
// CHECK-NOT: tt.annotate
