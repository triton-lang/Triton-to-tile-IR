// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=false flush-to-zero-modifier=false num-warps-in-cta=4},reconcile-unrealized-casts)" | FileCheck --check-prefixes=CHECK,NOFTZ %s
// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=false flush-to-zero-modifier=true num-warps-in-cta=4},reconcile-unrealized-casts)" | FileCheck --check-prefixes=CHECK,FTZ %s

// An explicit approximate division keeps its rounding mode regardless of the
// approx modifier; a plain division in the same kernel stays full range.
module {
  tt.func public @approx_divf(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %x = tt.load %a : !tt.ptr<f32>
    %y = tt.load %b : !tt.ptr<f32>
    %q = tt.approx_divf %x, %y : f32
    tt.store %out, %q : !tt.ptr<f32>
    %r = arith.divf %x, %y : f32
    tt.store %out, %r : !tt.ptr<f32>
    tt.return
  }
}
// CHECK-LABEL: entry @approx_divf
// NOFTZ: divf {{.*}}rounding<approx> : tile<f32>
// NOFTZ: divf {{.*}}rounding<full> : tile<f32>
// FTZ: divf {{.*}}rounding<approx> flush_to_zero : tile<f32>
// FTZ: divf {{.*}}rounding<full> flush_to_zero : tile<f32>
// CHECK-NOT: tt.approx_divf
