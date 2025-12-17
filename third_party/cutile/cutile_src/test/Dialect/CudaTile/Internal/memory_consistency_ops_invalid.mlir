// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

cuda_tile.module @invalid_fence_weak {
  entry @fence_invalid_memory_ordering() {
    // expected-error @below{{expect one of: relaxed, release, acquire, or acq_rel, but got: weak}}
    internal$fence_ordered weak device
    return
  }
}

// -----

cuda_tile.module @ordered_store {
  experimental$func @ordered_store(%ptr: !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, %val: !cuda_tile.tile<16x32xf32>) {
    // expected-error@below {{weak store must not have memory scope}}
    internal$store_ordered weak device %ptr, %val
      : !cuda_tile.tile<16x32x!cuda_tile.ptr<f32>>, !cuda_tile.tile<16x32xf32>
    return
  }
}