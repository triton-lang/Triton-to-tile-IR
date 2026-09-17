// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(cuda_tile.module(cuda_tile.entry(auto-gen-memory-token{autogen-alias-memtoken=true})))" | FileCheck %s

// Release publishes writes on other roots; acquire orders later reads of them.
module {
  cuda_tile.module @m {
    entry @message(%data: tile<ptr<i32>>, %flag: tile<ptr<i32>>) {
      %one = cuda_tile.constant <i32: 1> : tile<i32>
      %write = cuda_tile.store_ptr_tko weak %data, %one : tile<ptr<i32>>, tile<i32> -> token
      %publish = cuda_tile.store_ptr_tko release device %flag, %one : tile<ptr<i32>>, tile<i32> -> token
      %ready, %acquire = cuda_tile.load_ptr_tko acquire device %flag : tile<ptr<i32>> -> tile<i32>, token
      %value, %read = cuda_tile.load_ptr_tko weak %data : tile<ptr<i32>> -> tile<i32>, token
      cuda_tile.return
    }
  }
}
// CHECK-LABEL: @message
// CHECK: %[[WRITE:.*]] = store_ptr_tko weak
// CHECK: %[[PUBLISH:.*]] = store_ptr_tko release device {{.*}} token{{ ?}}={{ ?}}%[[WRITE]]
// CHECK: %{{.*}}, %[[ACQUIRE:.*]] = load_ptr_tko acquire device {{.*}} token{{ ?}}={{ ?}}%[[PUBLISH]]
// CHECK: %[[JOIN:.*]] = join_tokens %[[WRITE]], %[[ACQUIRE]]
// CHECK: load_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[JOIN]]

// -----

// A read-only loop must preserve its token for a following release on a
// different root. Dropping that loop token would let the release move earlier.
module {
  cuda_tile.module @m {
    entry @read_before_release(%data: tile<ptr<i32>>, %flag: tile<ptr<i32>>) {
      %zero = cuda_tile.constant <i32: 0> : tile<i32>
      %one = cuda_tile.constant <i32: 1> : tile<i32>
      %two = cuda_tile.constant <i32: 2> : tile<i32>
      cuda_tile.for %i in (%zero to %two, step %one) : tile<i32> {
        %v, %read = cuda_tile.load_ptr_tko weak %data : tile<ptr<i32>> -> tile<i32>, token
        cuda_tile.continue
      }
      %publish = cuda_tile.store_ptr_tko release device %flag, %one : tile<ptr<i32>>, tile<i32> -> token
      cuda_tile.return
    }
  }
}
// CHECK-LABEL: @read_before_release
// CHECK: %[[LOOP:.*]] = for
// CHECK: load_ptr_tko weak
// CHECK: continue {{.*}}token
// CHECK: store_ptr_tko release device {{.*}} token{{ ?}}={{ ?}}%[[LOOP]]
