// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt | FileCheck %s

// Note: Round-trip testing skipped - experimental pws ops are not supported in bytecode writer.

cuda_tile.module @kernels {

    entry @test_experimental_create_queue() {
        // CHECK: experimental$create_queue  : <[!cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>]>
        %0 = cuda_tile.experimental$create_queue  : <[
            !cuda_tile.tile<1024xf32>,
            !cuda_tile.tile<1024xf32>
        ]>
        // CHECK: experimental$create_queue  : <[!cuda_tile.tile<1024xf32>]>
        %1 = cuda_tile.experimental$create_queue  : <[!cuda_tile.tile<1024xf32>]>
    }

    experimental$func @test_experimental_queue_get(
        %arg0: !cuda_tile.queue<[!cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>]>
    ) -> (!cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>) {
        // CHECK: experimental$queue.get %{{.*}} {
        // CHECK: ^bb0(%{{.*}}: !cuda_tile.tile<1024xf32>, %{{.*}}: !cuda_tile.tile<1024xf32>):
        // CHECK: experimental$queue.yield %{{.*}}, %{{.*}} : !cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>
        // CHECK: } : !cuda_tile.queue<[!cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>]> ->
        // CHECK-SAME: !cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>
        %0, %1 = cuda_tile.experimental$queue.get %arg0 {
        ^bb0(%arg1: !cuda_tile.tile<1024xf32>, %arg2: !cuda_tile.tile<1024xf32>):
            cuda_tile.experimental$queue.yield %arg1, %arg2 :
                !cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>
        } : !cuda_tile.queue<[!cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>]> ->
            !cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>
        return %0, %1 : !cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>
    }

    experimental$func @test_experimental_queue_put(
        %arg0: !cuda_tile.queue<[!cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>]>,
        %arg1: !cuda_tile.tile<1024xf32>,
        %arg2: !cuda_tile.tile<1024xf32>
    ) {
        // CHECK: experimental$queue.put %{{.*}} {
        // CHECK: ^bb0(%{{.*}}: !cuda_tile.tile<1024xf32>, %{{.*}}: !cuda_tile.tile<1024xf32>):
        // CHECK: experimental$queue.yield %{{.*}}, %{{.*}} : !cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>
        // CHECK: } : !cuda_tile.queue<[!cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>]>
        cuda_tile.experimental$queue.put %arg0 {
        ^bb0(%arg3: !cuda_tile.tile<1024xf32>, %arg4: !cuda_tile.tile<1024xf32>):
            cuda_tile.experimental$queue.yield %arg1, %arg2 :
                !cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>
        } : !cuda_tile.queue<[!cuda_tile.tile<1024xf32>, !cuda_tile.tile<1024xf32>]>
        return
    }

    entry @test_experimental_execute() {
        %value = constant <f32: 42.0> : !cuda_tile.tile<f32>

        // CHECK: experimental$execute
        // CHECK-SAME: <{num_agents_per_group = array<i32: 1>}>
        // CHECK: ^bb0(%{{.*}}: !cuda_tile.tile<f32>):
        // CHECK: })
        "cuda_tile.experimental$execute"(%value) <{num_agents_per_group = array<i32: 1>}> ({
        ^bb0(%arg0: !cuda_tile.tile<f32>):
            %result = mulf %arg0, %arg0 : tile<f32>
        }) : (!cuda_tile.tile<f32>) -> ()
    }

    experimental$func @test_experimental_mark_for_reuse(
        %q0: !cuda_tile.queue<[!cuda_tile.tile<128x64xf16>]>,
        %q1: !cuda_tile.queue<[!cuda_tile.tile<128x64xf16>]>
    ) {
        // CHECK: experimental$mark_for_reuse %{{.*}}, %{{.*}} <{partitions = array<i32: 1, 1>}> :
        // CHECK-SAME: !cuda_tile.queue<[!cuda_tile.tile<128x64xf16>]>,
        // CHECK-SAME: !cuda_tile.queue<[!cuda_tile.tile<128x64xf16>]>
        cuda_tile.experimental$mark_for_reuse %q0, %q1 <{partitions = array<i32: 1, 1>}> :
            !cuda_tile.queue<[!cuda_tile.tile<128x64xf16>]>,
            !cuda_tile.queue<[!cuda_tile.tile<128x64xf16>]>
        return
    }

    entry @test_experimental_cancel_next_program_id() {
        // CHECK: %{{.*}} = experimental$cancel_next_program_id : <program_id>
        %program_id = cuda_tile.experimental$cancel_next_program_id : <program_id>
    }

    entry @test_experimental_get_program_id() {
        %program_id = cuda_tile.experimental$cancel_next_program_id : <program_id>
        // CHECK: %{{.*}} = experimental$get_program_id %{{.*}} axis = x : <program_id>
        %idx_x = cuda_tile.experimental$get_program_id %program_id axis = x : <program_id>
        // CHECK: %{{.*}} = experimental$get_program_id %{{.*}} axis = y : <program_id>
        %idx_y = cuda_tile.experimental$get_program_id %program_id axis = y : <program_id>
        // CHECK: %{{.*}} = experimental$get_program_id %{{.*}} axis = z : <program_id>
        %idx_z = cuda_tile.experimental$get_program_id %program_id axis = z : <program_id>
    }

    entry @test_experimental_is_valid_program_id() {
        %program_id = cuda_tile.experimental$cancel_next_program_id : <program_id>
        // CHECK: %{{.*}} = experimental$is_valid_program_id %{{.*}} : <program_id>
        %valid = cuda_tile.experimental$is_valid_program_id %program_id : <program_id>
    }

} // end module