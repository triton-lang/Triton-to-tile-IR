// RUN: %round_trip_test %s %t

// Test bytecode serialization/deserialization of producer attribute on ModuleOp.
// The producer attribute is optional and contains free-form text identifying
// what tool generated the bytecode (e.g., compiler version, build options).

cuda_tile.module @kernels attributes {producer = "nvcc version 13.3 -O2 --gpu-architecture=sm_90"} {
  cuda_tile.entry @simple_kernel(%a: !cuda_tile.tile<f32>) {
    cuda_tile.return
  }
}
