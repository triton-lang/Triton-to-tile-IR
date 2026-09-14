// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=true num-warps-in-cta=8},reconcile-unrealized-casts)" | FileCheck --check-prefix=APPROX %s
// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-triton-to-cuda-tile{approx-modifier=false num-warps-in-cta=4},reconcile-unrealized-casts)" | FileCheck --check-prefix=FULL %s

module {
  tt.func public @exp_precision(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %x = tt.load %input : !tt.ptr<f32>
    %value = math.exp %x : f32
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }
}
// APPROX-LABEL: entry @exp_precision
// APPROX-SAME: num_worker_warps_per_cta = 8
// APPROX: exp {{.*}} rounding<approx>
// FULL-LABEL: entry @exp_precision
// FULL-SAME: num_worker_warps_per_cta = 4
// FULL: exp {{.*}} rounding<full>
