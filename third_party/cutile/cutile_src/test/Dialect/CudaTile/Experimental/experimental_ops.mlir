// RUN: cuda-tile-opt %s | cuda-tile-opt | FileCheck %s
// RUN: cuda-tile-opt -mlir-print-op-generic %s | cuda-tile-opt | FileCheck %s
// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
    
    entry @test_experimental_log10() {
        // CHECK: %[[c_tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
        %c_tensor = constant <f32: [[1.0, 2.0], [4.0, 5.0]]> : !cuda_tile.tile<2x2xf32>
        // CHECK: experimental$log10 %[[c_tensor]] : tile<2x2xf32>
        %log10_1 = cuda_tile.experimental$log10 %c_tensor : tile<2x2xf32>
    }

    entry @test_experimental_log1p() {
        // CHECK: %[[c_tensor:.*]] = constant <f32: {{\[}}[1.000000e+00, 2.000000e+00], [4.000000e+00, 5.000000e+00]]> : tile<2x2xf32>
        %c_tensor = constant <f32: [[1.0, 2.0], [4.0, 5.0]]> : !cuda_tile.tile<2x2xf32>
        // CHECK: experimental$log1p %[[c_tensor]] : tile<2x2xf32>
        %log1p_1 = cuda_tile.experimental$log1p %c_tensor : tile<2x2xf32>
    }

    entry @test_experimental_generate() {
        // CHECK: experimental$generate
        %generated = cuda_tile.experimental$generate {
            // CHECK: ^{{.*}}(%[[ARG0:.*]]: !cuda_tile.tile<i32>, %[[ARG1:.*]]: !cuda_tile.tile<i32>):
            ^bb0(%arg0: !cuda_tile.tile<i32>, %arg1: !cuda_tile.tile<i32>):
            // CHECK: %[[RESULT:.*]] = addi %[[ARG0]], %[[ARG1]] : tile<i32>
            %0 = addi %arg0, %arg1 : tile<i32>
            // CHECK: yield %[[RESULT]] : tile<i32>
            yield %0 : tile<i32>
            // CHECK: tile<16x16xi32>
        } : tile<16x16xi32>
    }

    entry @test_experimental_inject_ir() {
        // CHECK: experimental$inject_ir {ir = "gpu.printf \22World\22", stage = "llvm"}
        experimental$inject_ir {stage = "llvm", ir = "gpu.printf \"World\""} {}
    }

    // CHECK: direct_callsite_fn
    experimental$func @direct_callsite_fn(%arg0: !cuda_tile.tile<2xi16>) {}

    // CHECK: direct_callsite_fn_multi_operands
    experimental$func @direct_callsite_fn_multi_operands(%arg0: !cuda_tile.tile<2xi16>, %arg1: !cuda_tile.tile<2xf32>) {}

    // CHECK: direct_callsite_fn_with_results
    experimental$func @direct_callsite_fn_with_results() -> !cuda_tile.tile<2xf32> {
      %0 = constant <f32: 2.0> : !cuda_tile.tile<2xf32>
      return %0 : !cuda_tile.tile<2xf32>
    }

    // CHECK: direct_callsite_fn_multi_operands_and_results
    experimental$func @direct_callsite_fn_multi_operands_and_results(%arg0: !cuda_tile.tile<2xi16>, %arg1: !cuda_tile.tile<2xf32>) -> (!cuda_tile.tile<2xf32>, !cuda_tile.tile<2xi16>)  {
      return %arg1, %arg0 : !cuda_tile.tile<2xf32>, !cuda_tile.tile<2xi16>
    }


    experimental$func @test_experimental_call(%arg0: !cuda_tile.tile<2xi16>, %arg1: !cuda_tile.tile<2xf32>) {
      // CHECK: experimental$call @direct_callsite_fn
      cuda_tile.experimental$call @direct_callsite_fn(%arg0) : (!cuda_tile.tile<2xi16>) -> ()
      // CHECK: experimental$call @direct_callsite_fn_multi_operands
      cuda_tile.experimental$call @direct_callsite_fn_multi_operands(%arg0, %arg1) : (!cuda_tile.tile<2xi16>, !cuda_tile.tile<2xf32>) -> ()
      // CHECK: experimental$call @direct_callsite_fn_with_results
      %0 = cuda_tile.experimental$call @direct_callsite_fn_with_results() : () -> (!cuda_tile.tile<2xf32>)

      // CHECK: experimental$call @direct_callsite_fn_multi_operands_and_results
      %1:2 = cuda_tile.experimental$call @direct_callsite_fn_multi_operands_and_results(%arg0, %arg1) : (!cuda_tile.tile<2xi16>, !cuda_tile.tile<2xf32>) -> (!cuda_tile.tile<2xf32>, !cuda_tile.tile<2xi16>)
    }


    experimental$func @test_experimental_histogram(%arg0: !cuda_tile.tile<512xi32>) {
    // CHECK: experimental$histogram %{{.+}} : tile<512xi32> -> tile<16xi32>
    %1 = cuda_tile.experimental$histogram %arg0 : tile<512xi32> -> tile<16xi32>
    }

    experimental$func @test_experimental_elementwise_inline_asm(%arg0: !cuda_tile.tile<512xi8>) {
    // CHECK: experimental$elementwise_inline_asm "shl.b32 $0, $0, 3;"
    %0 = cuda_tile.experimental$elementwise_inline_asm "shl.b32 $0, $0, 3;"
        {constraints = "=r,r", packed_element = 4 : i32, pure = true} 
        %arg0 : !cuda_tile.tile<512xi8> -> !cuda_tile.tile<512xi8>
    }

    entry @test_experimental_elementwise_inline_asm_globaltimer() {
    // CHECK: experimental$elementwise_inline_asm "mov.u64 $0, %globaltimer;"
    %0 = cuda_tile.experimental$elementwise_inline_asm "mov.u64 $0, %globaltimer;"
        {constraints = "=l", packed_element = 1 : i32, pure = false} 
        -> !cuda_tile.tile<i64>
    }

    experimental$func @test_experimental_pragma(%arg0: !cuda_tile.tile<1x2x4xf32>, %arg1: !cuda_tile.tile<4xf32>, %arg2: i32) {
        // CHECK: experimental$pragma {
        // CHECK-NEXT: }
        cuda_tile.experimental$pragma {
            cuda_tile.yield
        }

        // CHECK: experimental$pragma {
        // CHECK-NEXT: cuda_tile.yield %arg0 : tile<1x2x4xf32>
        // CHECK-NEXT: } <{ocgEnterDirectives = dense<[".pragma \22set knob SchedResBusyMachineOpcode=FMMA2,4+MUFU.EX2,3\22;",
        // CHECK-SAME: ".pragma \22next knob FenceCode\22;"]> : tensor<2x!cuda_tile.string>,
        // CHECK-SAME: ocgLeaveDirectives = dense<".pragma \22reset knob SchedResBusyMachineOpcode\22;"> : tensor<1x!cuda_tile.string>}>
        // CHECK-SAME: : !cuda_tile.tile<1x2x4xf32>
        %0 = cuda_tile.experimental$pragma {
            cuda_tile.yield %arg0 : !cuda_tile.tile<1x2x4xf32>
        } <{
            ocgEnterDirectives = dense<[".pragma \"set knob SchedResBusyMachineOpcode=FMMA2,4+MUFU.EX2,3\";",
                                        ".pragma \"next knob FenceCode\";"]> : tensor<2x!cuda_tile.string>,
            ocgLeaveDirectives = dense<[".pragma \"reset knob SchedResBusyMachineOpcode\";"]> : tensor<1x!cuda_tile.string>
            }> : !cuda_tile.tile<1x2x4xf32>

        // CHECK: experimental$pragma {
        // CHECK-NEXT: cuda_tile.yield %arg1, %arg0 : tile<4xf32>, tile<1x2x4xf32>
        // CHECK-NEXT: } : !cuda_tile.tile<4xf32>, !cuda_tile.tile<1x2x4xf32>
        %1:2 = cuda_tile.experimental$pragma {
            cuda_tile.yield %arg1, %arg0 : !cuda_tile.tile<4xf32>, !cuda_tile.tile<1x2x4xf32>
        } : !cuda_tile.tile<4xf32>, !cuda_tile.tile<1x2x4xf32>

        // CHECK: experimental$pragma {
        // CHECK-NEXT: cuda_tile.yield %arg2 : i32
        // CHECK-NEXT: } : i32
        %3 = cuda_tile.experimental$pragma {
            cuda_tile.yield %arg2 : i32
        } : i32
    }

    // CHECK: experimental$func
    cuda_tile.experimental$func @experimental$func(%arg0: !cuda_tile.tile<2x2xf32>) {}

    // CHECK: experimental$func @foo() {
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    "cuda_tile.experimental$func"() ({
    ^bb0:
        return
    }) {function_type = () -> (), sym_name = "foo"} : () -> ()

    // CHECK: experimental$func @foo2() {
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    experimental$func @foo2() {}

    // CHECK: experimental$func @foo3(%{{.+}}: tile<2x2xf32>) {
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    cuda_tile.experimental$func @foo3(%arg0: !cuda_tile.tile<2x2xf32>) {}
    // CHECK-NEXT: experimental$func @foo1(
    // CHECK-SAME: %{{.+}}: tile<4x2xf32>, %{{.+}}: tile<4x2xf32>)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
    cuda_tile.experimental$func @foo1(%arg0: !cuda_tile.tile<4x2xf32>, %arg1: !cuda_tile.tile<4x2xf32>) {}


    // CHECK: experimental$func @func_early_exit
    experimental$func @func_early_exit() {
        %c1 = constant <i1: true> : !cuda_tile.tile<i1>

        // CHECK: if
        if %c1 {
        if %c1 {
            // CHECK: return
            return
        } else {
            // CHECK: return
            return
        }
        // CHECK: return
        return
        }
    }

    // CHECK-LABEL: @experimental$func_with_result
    // CHECK-SAME: %[[ARG0:.+]]: tile<2x2xf32>) -> tile<2x2xf32>
    experimental$func @experimental$func_with_result(%arg0: !cuda_tile.tile<2x2xf32>) -> !cuda_tile.tile<2x2xf32> {
        // CHECK: return %[[ARG0]] : tile<2x2xf32>
        return %arg0 : tile<2x2xf32>
    }

    // CHECK: experimental$func @experimental$func_with_kernel_scope_device
    experimental$func @experimental$func_with_kernel_scope_device() {}
    
    // CHECK-LABEL: entry @cancel_tile_block()
    entry @cancel_tile_block() {
        // CHECK: experimental$cancel_next_tile_block
        %x, %y, %z, %status = experimental$cancel_next_tile_block
    }

    experimental$func @test_experimental_gather_load(
        %arg0: !cuda_tile.tensor_view<1024x1024xf32, strides=[1024,1]>, 
        %arg1: !cuda_tile.tile<128xi32>, 
        %arg2: !cuda_tile.tile<i32>
    ) {
        // CHECK: experimental$gather_load %arg0 index = [%arg1]
        // CHECK: offset = [%arg2] <{dim = 0 : i64}> : 
        // CHECK: tensor_view<1024x1024xf32, strides=[1024,1]>,
        // CHECK: tile<128xi32>, !cuda_tile.tile<i32> -> tile<128x64xf32>
        %0 = experimental$gather_load %arg0 index = [%arg1]
        offset = [%arg2] < {dim = 0 : i64}> :
        !cuda_tile.tensor_view<1024x1024xf32, strides=[1024,1]>, 
        !cuda_tile.tile<128xi32>, !cuda_tile.tile<i32> -> !cuda_tile.tile<128x64xf32> 
    }

    experimental$func @test_experimental_scatter_store(
        %arg0: !cuda_tile.tile<128x64xf32>,
        %arg1: !cuda_tile.tensor_view<1024x1024xf32, strides=[1024,1]>,
        %arg2: !cuda_tile.tile<128xi32>,
        %arg3: !cuda_tile.tile<i32>,
        %arg4: !cuda_tile.token
    ) {
        // CHECK: experimental$scatter_store %arg0, %arg1, %arg2, [%arg3] token = %arg4 <{dim = 0 : i64}> :
        // CHECK: tile<128x64xf32>, tensor_view<1024x1024xf32, strides=[1024,1]>,
        // CHECK: tile<128xi32>, tile<i32>, token -> token
        %0 = experimental$scatter_store %arg0, %arg1, %arg2, [%arg3] token = %arg4 <{dim = 0 : i64}> :
        !cuda_tile.tile<128x64xf32>, !cuda_tile.tensor_view<1024x1024xf32, strides=[1024,1]>,
        !cuda_tile.tile<128xi32>, !cuda_tile.tile<i32>, !cuda_tile.token -> !cuda_tile.token
    }

    experimental$func @test_asin(
        %arg0 : !cuda_tile.tile<2xf32>,
        %arg1 : !cuda_tile.tile<2xf64>
    ) {
        // CHECK: experimental$asin %arg0 : tile<2xf32>
        %0 = experimental$asin %arg0 : tile<2xf32>
        // CHECK: experimental$asin %arg1 : tile<2xf64>
        %1 = experimental$asin %arg1 : tile<2xf64>
    }

    experimental$func @test_tiled_atomic_rmw(
        %arg0: !cuda_tile.partition_view<tile=(2x16),
               !cuda_tile.tensor_view<16x16xi32, strides=[16, 1]>>, 
        %arg1: !cuda_tile.tile<2x16xi32>) {
    %idx0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>
    %idx1 = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>
    // CHECK: experimental$tiled_atomic_rmw_unordered weak %arg0[%cst_0_i32, %cst_1_i32],umax, 
    // CHECK: %arg1  attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
    // CHECK: partition_view<tile=(2x16), tensor_view<16x16xi32, strides=[16,1]>>, tile<i32>, tile<2x16xi32> 
    // CHECK:-> !cuda_tile.tile<2x16xi32>, token
    %result_0, %result_token_0 = experimental$tiled_atomic_rmw_unordered 
                                    weak %arg0[%idx0, %idx1],
                                    umax, %arg1 attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
                                    : !cuda_tile.partition_view<tile=(2x16),
                                         !cuda_tile.tensor_view<16x16xi32, strides=[16, 1]>>, 
                                         tile<i32>,
                                         !cuda_tile.tile<2x16xi32> -> !cuda_tile.tile<2x16xi32>, token
    // CHECK: experimental$tiled_atomic_rmw_unordered weak %arg0[%cst_0_i32, %cst_1_i32],max, 
    // CHECK: %arg1  attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
    // CHECK: partition_view<tile=(2x16), tensor_view<16x16xi32, strides=[16,1]>>, tile<i32>, tile<2x16xi32> 
    // CHECK:-> !cuda_tile.tile<2x16xi32>, token
    %result_1, %result_token_1 = experimental$tiled_atomic_rmw_unordered 
                                    weak %arg0[%idx0, %idx1],
                                    max, %arg1 attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
                                    : !cuda_tile.partition_view<tile=(2x16),
                                         !cuda_tile.tensor_view<16x16xi32, strides=[16, 1]>>, 
                                         tile<i32>,
                                         !cuda_tile.tile<2x16xi32> -> !cuda_tile.tile<2x16xi32>, token
    // CHECK: experimental$tiled_atomic_rmw_unordered weak %arg0[%cst_0_i32, %cst_1_i32],umin, 
    // CHECK: %arg1  attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
    // CHECK: partition_view<tile=(2x16), tensor_view<16x16xi32, strides=[16,1]>>, tile<i32>, tile<2x16xi32> 
    // CHECK:-> !cuda_tile.tile<2x16xi32>, token
    %result_2, %result_token_2 = experimental$tiled_atomic_rmw_unordered 
                                    weak %arg0[%idx0, %idx1],
                                    umin, %arg1 attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
                                    : !cuda_tile.partition_view<tile=(2x16),
                                         !cuda_tile.tensor_view<16x16xi32, strides=[16, 1]>>, 
                                         tile<i32>,
                                         !cuda_tile.tile<2x16xi32> -> !cuda_tile.tile<2x16xi32>, token
    // CHECK: experimental$tiled_atomic_rmw_unordered weak %arg0[%cst_0_i32, %cst_1_i32],min, 
    // CHECK: %arg1  attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
    // CHECK: partition_view<tile=(2x16), tensor_view<16x16xi32, strides=[16,1]>>, tile<i32>, tile<2x16xi32> 
    // CHECK:-> !cuda_tile.tile<2x16xi32>, token
    %result_3, %result_token_3 = experimental$tiled_atomic_rmw_unordered 
                                    weak %arg0[%idx0, %idx1],
                                    min, %arg1 attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
                                    : !cuda_tile.partition_view<tile=(2x16),
                                         !cuda_tile.tensor_view<16x16xi32, strides=[16, 1]>>, 
                                         tile<i32>,
                                         !cuda_tile.tile<2x16xi32> -> !cuda_tile.tile<2x16xi32>, token
    // CHECK: experimental$tiled_atomic_rmw_unordered weak %arg0[%cst_0_i32, %cst_1_i32],add, 
    // CHECK: %arg1  attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
    // CHECK: partition_view<tile=(2x16), tensor_view<16x16xi32, strides=[16,1]>>, tile<i32>, tile<2x16xi32> 
    // CHECK:-> !cuda_tile.tile<2x16xi32>, token
    %result_4, %result_token_4 = experimental$tiled_atomic_rmw_unordered 
                                    weak %arg0[%idx0, %idx1],
                                    add, %arg1 attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
                                    : !cuda_tile.partition_view<tile=(2x16),
                                         !cuda_tile.tensor_view<16x16xi32, strides=[16, 1]>>, 
                                         tile<i32>,
                                         !cuda_tile.tile<2x16xi32> -> !cuda_tile.tile<2x16xi32>, token
    // CHECK: experimental$tiled_atomic_rmw_unordered weak %arg0[%cst_0_i32, %cst_1_i32],or, 
    // CHECK: %arg1  attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
    // CHECK: partition_view<tile=(2x16), tensor_view<16x16xi32, strides=[16,1]>>, tile<i32>, tile<2x16xi32> 
    // CHECK:-> !cuda_tile.tile<2x16xi32>, token
    %result_5, %result_token_5 = experimental$tiled_atomic_rmw_unordered 
                                    weak %arg0[%idx0, %idx1],
                                    or, %arg1 attributes {optimization_hints = #cuda_tile.optimization_hints<sm_100 = {latency = 3}>} 
                                    : !cuda_tile.partition_view<tile=(2x16),
                                         !cuda_tile.tensor_view<16x16xi32, strides=[16, 1]>>, 
                                         tile<i32>,
                                         !cuda_tile.tile<2x16xi32> -> !cuda_tile.tile<2x16xi32>, token
    }

    // CHECK-LABEL: test_extern_elementwise
    experimental$func @test_extern_elementwise(%arg0: !cuda_tile.tile<128xf64>) {
        // CHECK: experimental$extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__nv_cyl_bessel_i1"}
        %2 = experimental$extern_elementwise %arg0 
                        {libname = "", libpath = "", pure = true, symbol = "__nv_cyl_bessel_i1"} 
                        : (!cuda_tile.tile<128xf64>) -> !cuda_tile.tile<128xf64>
        return
    }
    
    // CHECK-LABEL: test_alloca_private
    experimental$func @test_alloca_private() {
      // CHECK: experimental$alloca num_elem = 64, alignment = 16 : tile<ptr<f32>>
      %0 = experimental$alloca num_elem = 64, alignment = 16 : tile<ptr<f32>>
    }

    // CHECK-LABEL: test_alloca_global
    experimental$func @test_alloca_global() {
      // CHECK: experimental$alloca num_elem = 64, alignment = 16 global : tile<ptr<f32>>
      %0 = experimental$alloca num_elem = 64, alignment = 16 global : tile<ptr<f32>>
    }

    // CHECK-LABEL: test_mmaf_scaled_fp8e5m2
    experimental$func @test_mmaf_scaled_fp8e5m2(
        %arg0: !cuda_tile.tile<128x128xf8E5M2>,
        %arg1: !cuda_tile.tile<128x128xf8E5M2>,
        %arg2: !cuda_tile.tile<128x128xf32>,
        %arg3: !cuda_tile.tile<128x4xf8E8M0FNU>,
        %arg4: !cuda_tile.tile<4x128xf8E8M0FNU>
    ) {
        // CHECK: experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
        // CHECK-SAME:     tile<128x128xf8E5M2>,
        // CHECK-SAME:     tile<128x128xf8E5M2>,
        // CHECK-SAME:     tile<128x128xf32>,
        // CHECK-SAME:     tile<128x4xf8E8M0FNU>,
        // CHECK-SAME:     tile<4x128xf8E8M0FNU>
        // CHECK-SAME:     -> tile<128x128xf32>
        %0 = experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
            !cuda_tile.tile<128x128xf8E5M2>,
            !cuda_tile.tile<128x128xf8E5M2>,
            !cuda_tile.tile<128x128xf32>,
            !cuda_tile.tile<128x4xf8E8M0FNU>,
            !cuda_tile.tile<4x128xf8E8M0FNU>
            -> !cuda_tile.tile<128x128xf32>
    }

    // CHECK-LABEL: test_mmaf_scaled_fp8e4m3
    experimental$func @test_mmaf_scaled_fp8e4m3(
        %arg0: !cuda_tile.tile<128x128xf8E4M3FN>,
        %arg1: !cuda_tile.tile<128x128xf8E4M3FN>,
        %arg2: !cuda_tile.tile<128x128xf32>,
        %arg3: !cuda_tile.tile<128x4xf8E8M0FNU>,
        %arg4: !cuda_tile.tile<4x128xf8E8M0FNU>
    ) {
        // CHECK: experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
        // CHECK-SAME:     tile<128x128xf8E4M3FN>,
        // CHECK-SAME:     tile<128x128xf8E4M3FN>,
        // CHECK-SAME:     tile<128x128xf32>,
        // CHECK-SAME:     tile<128x4xf8E8M0FNU>,
        // CHECK-SAME:     tile<4x128xf8E8M0FNU>
        // CHECK-SAME:     -> tile<128x128xf32>
        %0 = experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
            !cuda_tile.tile<128x128xf8E4M3FN>,
            !cuda_tile.tile<128x128xf8E4M3FN>,
            !cuda_tile.tile<128x128xf32>,
            !cuda_tile.tile<128x4xf8E8M0FNU>,
            !cuda_tile.tile<4x128xf8E8M0FNU>
            -> !cuda_tile.tile<128x128xf32>
    }

    // CHECK-LABEL: test_mmaf_scaled_mxfp4
    experimental$func @test_mmaf_scaled_mxfp4(
        %arg0: !cuda_tile.tile<128x128xf4E2M1FN>,
        %arg1: !cuda_tile.tile<128x128xf4E2M1FN>,
        %arg2: !cuda_tile.tile<128x128xf32>,
        %arg3: !cuda_tile.tile<128x4xf8E8M0FNU>,
        %arg4: !cuda_tile.tile<4x128xf8E8M0FNU>
    ) {
        // CHECK: experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
        // CHECK-SAME:     tile<128x128xf4E2M1FN>,
        // CHECK-SAME:     tile<128x128xf4E2M1FN>,
        // CHECK-SAME:     tile<128x128xf32>,
        // CHECK-SAME:     tile<128x4xf8E8M0FNU>,
        // CHECK-SAME:     tile<4x128xf8E8M0FNU>
        // CHECK-SAME:     -> tile<128x128xf32>
        %0 = experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
            !cuda_tile.tile<128x128xf4E2M1FN>,
            !cuda_tile.tile<128x128xf4E2M1FN>,
            !cuda_tile.tile<128x128xf32>,
            !cuda_tile.tile<128x4xf8E8M0FNU>,
            !cuda_tile.tile<4x128xf8E8M0FNU>
            -> !cuda_tile.tile<128x128xf32>
    }

    // CHECK-LABEL: test_mmaf_scaled_nvfp4_f4e2m1_f8e4m3
    experimental$func @test_mmaf_scaled_nvfp4_f4e2m1_f8e4m3(
        %arg0: !cuda_tile.tile<128x128xf4E2M1FN>,
        %arg1: !cuda_tile.tile<128x128xf4E2M1FN>,
        %arg2: !cuda_tile.tile<128x128xf32>,
        %arg3: !cuda_tile.tile<128x8xf8E4M3FN>,
        %arg4: !cuda_tile.tile<8x128xf8E4M3FN>
    ) {
        // CHECK: experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
        // CHECK-SAME:     tile<128x128xf4E2M1FN>,
        // CHECK-SAME:     tile<128x128xf4E2M1FN>,
        // CHECK-SAME:     tile<128x128xf32>,
        // CHECK-SAME:     tile<128x8xf8E4M3FN>,
        // CHECK-SAME:     tile<8x128xf8E4M3FN>
        // CHECK-SAME:     -> tile<128x128xf32>
        %0 = experimental$mmaf_scaled %arg0, %arg1, %arg2, %arg3, %arg4 :
            !cuda_tile.tile<128x128xf4E2M1FN>,
            !cuda_tile.tile<128x128xf4E2M1FN>,
            !cuda_tile.tile<128x128xf32>,
            !cuda_tile.tile<128x8xf8E4M3FN>,
            !cuda_tile.tile<8x128xf8E4M3FN>
            -> !cuda_tile.tile<128x128xf32>
    }

    // CHECK-LABEL: test_reinterpret
    experimental$func @test_reinterpret(
        %arg0: !cuda_tile.tile<128x64xi8>
    ) {
        // CHECK: experimental$reinterpret %arg0 : tile<128x64xi8> -> tile<128x128xf4E2M1FN>
        %0 = experimental$reinterpret %arg0 : !cuda_tile.tile<128x64xi8> -> !cuda_tile.tile<128x128xf4E2M1FN>
    }

    experimental$func @test_unpack_op(%arg0: !cuda_tile.tile<64xi8>) {
        // CHECK: experimental$unpack %{{.+}} : tile<64xi8> -> tile<128xf4E2M1FN>
        %0 = experimental$unpack %arg0 : tile<64xi8> -> tile<128xf4E2M1FN>
    }

    experimental$func @test_pack_op(%arg_i16: !cuda_tile.tile<64xi16>, %arg_f32: !cuda_tile.tile<64xf32>) {
        // CHECK: experimental$pack %{{.+}} : tile<64xi16> -> tile<128xi8>
        %0 = experimental$pack %arg_i16 : tile<64xi16> -> tile<128xi8>
        // CHECK: experimental$pack %{{.+}} : tile<64xf32> -> tile<256xi8>
        %1 = experimental$pack %arg_f32 : tile<64xf32> -> tile<256xi8>
    }

    // CHECK-LABEL: test_make_strided_view
    // CHECK-SAME: (%[[TENSOR_VIEW:.+]]: tensor_view<8192x8192x64xf32, strides=[524288,64,1]>,
    // CHECK-SAME: %[[TENSOR_VIEW_DYN:.+]]: tensor_view<?x8192x64xf32, strides=[?,64,1]>)
    experimental$func @test_make_strided_view(%tensor_view: !cuda_tile.tensor_view<8192x8192x64xf32, strides=[524288,64,1]>,
                                              %tensor_view_dyn: !cuda_tile.tensor_view<?x8192x64xf32, strides=[?,64,1]>) {
        // CHECK: experimental$make_strided_view %[[TENSOR_VIEW]] : strided_view<tile=(1x1x1), traversal_strides=[1,1,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
        experimental$make_strided_view %tensor_view : strided_view<tile=(1x1x1), traversal_strides=[1,1,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>

        // CHECK: experimental$make_strided_view %[[TENSOR_VIEW]] : strided_view<tile=(1x1x1), traversal_strides=[1,1,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
        experimental$make_strided_view %tensor_view : strided_view<tile=(1x1x1), traversal_strides=[1,1,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
        // CHECK: experimental$make_strided_view %[[TENSOR_VIEW]] : strided_view<tile=(1024x8192x2), traversal_strides=[512,8192,2], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
        experimental$make_strided_view %tensor_view : strided_view<tile=(1024x8192x2), traversal_strides=[512,8192,2], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
        // CHECK: experimental$make_strided_view %[[TENSOR_VIEW]] : strided_view<tile=(1024x8x1024), traversal_strides=[512,8,512], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>, dim_map=[0, 2, 1]>
        experimental$make_strided_view %tensor_view : strided_view<tile=(1024x8x1024), traversal_strides=[512,8,512], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>, dim_map=[0, 2, 1]>

        // CHECK: experimental$make_strided_view %[[TENSOR_VIEW_DYN]] : strided_view<tile=(1x1x1), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>>
        experimental$make_strided_view %tensor_view_dyn : strided_view<tile=(1x1x1), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>>
        // CHECK: experimental$make_strided_view %[[TENSOR_VIEW_DYN]] : strided_view<tile=(1024x8192x2), traversal_strides=[1024,8192,2], tensor_view<?x8192x64xf32, strides=[?,64,1]>>
        experimental$make_strided_view %tensor_view_dyn : strided_view<tile=(1024x8192x2), traversal_strides=[1024,8192,2], tensor_view<?x8192x64xf32, strides=[?,64,1]>>
        // CHECK: experimental$make_strided_view %[[TENSOR_VIEW_DYN]] : strided_view<tile=(1024x8x1024), traversal_strides=[1024,8,1024], tensor_view<?x8192x64xf32, strides=[?,64,1]>, dim_map=[0, 2, 1]>
        experimental$make_strided_view %tensor_view_dyn : strided_view<tile=(1024x8x1024), traversal_strides=[1024,8,1024], tensor_view<?x8192x64xf32, strides=[?,64,1]>, dim_map=[0, 2, 1]>
    }

    // CHECK-LABEL: get_index_space_shape_strided_view
    // CHECK-SAME: (%[[VIEW:.*]]: strided_view<tile=(8x1x16), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>>)
    experimental$func @get_index_space_shape_strided_view(%strided_view: !cuda_tile.strided_view<tile=(8x1x16), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>>) {
        // CHECK: %[[SIZE_I32:.*]]:3 = get_index_space_shape %[[VIEW]] : strided_view<tile=(8x1x16), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i32>
        %size_i32:3 = get_index_space_shape %strided_view : strided_view<tile=(8x1x16), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i32>

        // CHECK: %[[SIZE_I16:.*]]:3 = get_index_space_shape %[[VIEW]] : strided_view<tile=(8x1x16), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i16>
        %size_i16:3 = get_index_space_shape %strided_view : strided_view<tile=(8x1x16), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i16>

        // CHECK: %[[SIZE_I64:.*]]:3 = get_index_space_shape %[[VIEW]] : strided_view<tile=(8x1x16), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i64>
        %size_i64:3 = get_index_space_shape %strided_view : strided_view<tile=(8x1x16), traversal_strides=[1,1,1], tensor_view<?x8192x64xf32, strides=[?,64,1]>> -> tile<i64>
    }

    // CHECK-LABEL: load_store_tile_strided_view
    // CHECK-SAME: (%[[VIEW1:.+]]: strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>
    // CHECK-SAME:  %[[VIEW3:.+]]: strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>
    // CHECK-SAME:  %[[T1:.+]]: tile<8xf32>, %[[T3:.+]]: tile<1024x1024x8xf32>
    experimental$func @load_store_tile_strided_view(%view1: !cuda_tile.strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>,
                                                    %view3: !cuda_tile.strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>,
                                                    %t1: !cuda_tile.tile<8xf32>, %t3: !cuda_tile.tile<1024x1024x8xf32>) {
        // CHECK: %[[C0I64:.+]] = constant <i64: 0> : tile<i64>
        %c0i64 = constant <i64: 0> : !cuda_tile.tile<i64>
        // CHECK: %[[C0I32:.+]] = constant <i32: 0> : tile<i32>
        %c0i32 = constant <i32: 0> : !cuda_tile.tile<i32>
        // CHECK: %[[C0I16:.+]] = constant <i16: 0> : tile<i16>
        %c0i16 = constant <i16: 0> : !cuda_tile.tile<i16>
        // CHECK: %[[C0I8:.+]] = constant <i8: 0> : tile<i8>
        %c0i8 = constant <i8: 0> : !cuda_tile.tile<i8>
        // CHECK: %[[C0I1:.+]] = constant <i1: false> : tile<i1>
        %c0i1 = constant <i1: false> : !cuda_tile.tile<i1>

        // Stores

        // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I64]]] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i64> -> token
        // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I64]], %[[C0I64]], %[[C0I64]]] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i64> -> token
        %s1i64 = store_view_tko weak %t1, %view1[%c0i64] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i64> -> token
        %s2i64 = store_view_tko weak %t3, %view3[%c0i64, %c0i64, %c0i64] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i64> -> token
    
        // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I32]]] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
        // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token
        %s1i32 = store_view_tko weak %t1, %view1[%c0i32] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i32> -> token
        %s2i32 = store_view_tko weak %t3, %view3[%c0i32, %c0i32, %c0i32] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> token
    
        // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I16]]] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i16> -> token
        // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I16]], %[[C0I16]], %[[C0I16]]] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i16> -> token
        %s1i16 = store_view_tko weak %t1, %view1[%c0i16] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i16> -> token
        %s2i16 = store_view_tko weak %t3, %view3[%c0i16, %c0i16, %c0i16] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i16> -> token

        // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I8]]] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i8> -> token
        // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I8]], %[[C0I8]], %[[C0I8]]] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i8> -> token
        %s1i8 = store_view_tko weak %t1, %view1[%c0i8] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i8> -> token
        %s2i8 = store_view_tko weak %t3, %view3[%c0i8, %c0i8, %c0i8] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i8> -> token

        // CHECK: %{{.+}} = store_view_tko weak %[[T1]], %[[VIEW1]][%[[C0I1]]] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i1> -> token
        // CHECK: %{{.+}} = store_view_tko weak %[[T3]], %[[VIEW3]][%[[C0I1]], %[[C0I1]], %[[C0I1]]] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i1> -> token
        %s1i1 = store_view_tko weak %t1, %view1[%c0i1] : tile<8xf32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i1> -> token
        %s2i1 = store_view_tko weak %t3, %view3[%c0i1, %c0i1, %c0i1] : tile<1024x1024x8xf32>, strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i1> -> token

        // Loads

        // CHECK: %[[T1_I64:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I64]]] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i64> -> tile<8xf32>, token
        // CHECK: %[[T3_I64:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I64]], %[[C0I64]], %[[C0I64]]] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i64> -> tile<1024x1024x8xf32>, token
        %t1i64, %tok0i64 = load_view_tko weak %view1[%c0i64] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i64> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
        %t3i64, %tok1i64 = load_view_tko weak %view3[%c0i64, %c0i64, %c0i64] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i64> -> tile<1024x1024x8xf32>, token

        // CHECK: %[[T1_I32:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I32]]] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i32> -> tile<8xf32>, token
        // CHECK: %[[T3_I32:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I32]], %[[C0I32]], %[[C0I32]]] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token
        %t1i32, %tok0i32 = load_view_tko weak %view1[%c0i32] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i32> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
        %t3i32, %tok1i32 = load_view_tko weak %view3[%c0i32, %c0i32, %c0i32] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i32> -> tile<1024x1024x8xf32>, token

        // CHECK: %[[T1_I16:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I16]]] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i16> -> tile<8xf32>, token
        // CHECK: %[[T3_I16:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I16]], %[[C0I16]], %[[C0I16]]] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i16> -> tile<1024x1024x8xf32>, token
        %t1i16, %tok0i16 = load_view_tko weak %view1[%c0i16] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i16> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
        %t3i16, %tok1i16 = load_view_tko weak %view3[%c0i16, %c0i16, %c0i16] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i16> -> tile<1024x1024x8xf32>, token

        // CHECK: %[[T1_I8:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I8]]] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i8> -> tile<8xf32>, token
        // CHECK: %[[T3_I8:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I8]], %[[C0I8]], %[[C0I8]]] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i8> -> tile<1024x1024x8xf32>, token
        %t1i8, %tok0i8 = load_view_tko weak %view1[%c0i8] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i8> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
        %t3i8, %tok1i8 = load_view_tko weak %view3[%c0i8, %c0i8, %c0i8] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i8> -> tile<1024x1024x8xf32>, token

        // CHECK: %[[T1_I1:.+]], %{{.+}} = load_view_tko weak %[[VIEW1]][%[[C0I1]]] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i1> -> tile<8xf32>, token
        // CHECK: %[[T3_I1:.+]], %{{.+}} = load_view_tko weak %[[VIEW3]][%[[C0I1]], %[[C0I1]], %[[C0I1]]] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i1> -> tile<1024x1024x8xf32>, token
        %t1i1, %tok0i1 = load_view_tko weak %view1[%c0i1] : strided_view<tile=(8), traversal_strides=[1], tensor_view<128xf32, strides=[1]>>, tile<i1> -> !cuda_tile.tile<8xf32>, !cuda_tile.token
        %t3i1, %tok1i1 = load_view_tko weak %view3[%c0i1, %c0i1, %c0i1] : strided_view<tile=(1024x1024x8), traversal_strides=[1024,1024,1], tensor_view<8192x8192x64xf32, strides=[524288,64,1]>>, tile<i1> -> tile<1024x1024x8xf32>, token
    }
} // end module
