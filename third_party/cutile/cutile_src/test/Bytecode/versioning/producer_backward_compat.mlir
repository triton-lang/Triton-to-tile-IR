// Test that producer attribute is silently dropped when targeting older bytecode versions.
// The producer section is only available in version 13.3+.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 %s -o %t.bc
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.bc | FileCheck %s

// CHECK: cuda_tile.module @kernels
// CHECK-NOT: producer

cuda_tile.module @kernels attributes {producer = "test-producer v1.0"} {
  cuda_tile.entry @simple_kernel(%a: !cuda_tile.tile<f32>) {
    cuda_tile.return
  }
}
