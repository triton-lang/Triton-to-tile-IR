// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

// ****************** cuda_tile.experimental$erf ******************

cuda_tile.module @test_erf {
  experimental$func @kernel(%arg0 : !cuda_tile.tile<4xi16>) {
    // expected-error @below{{must be tile of f32 or f64 values, but got '!cuda_tile.tile<4xi16>'}}
    %0 = cuda_tile.experimental$erf %arg0 : !cuda_tile.tile<4xi16>
  }
}

// -----

cuda_tile.module @test_erf {
  experimental$func @kernel(%arg0 : !cuda_tile.tile<4xf16>) {
    // expected-error @below{{must be tile of f32 or f64 values, but got '!cuda_tile.tile<4xf16>'}}
    %0 = cuda_tile.experimental$erf %arg0 : !cuda_tile.tile<4xf16>
  }
}

// -----

cuda_tile.module @erf_different_element_type {// expected-note @below{{prior use here}}
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.experimental$erf %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @erf_different_shape {// expected-note @below{{prior use here}}
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.experimental$erf %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @erf_different_rank {// expected-note @below{{prior use here}}
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.experimental$erf %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @erf_invalid_type_i32 {
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.experimental$erf' op operand #0 must be tile of f32 or f64 values}}
        %0 = cuda_tile.experimental$erf %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

// ****************** cuda_tile.experimental$log10 ******************

cuda_tile.module @log10_different_element_type_type {// expected-note @below{{prior use here}}
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.experimental$log10 %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @log10_different_shape {// expected-note @below{{prior use here}}
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.experimental$log10 %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @log10_different_rank {// expected-note @below{{prior use here}}
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.experimental$log10 %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @log10_invalid_type_i32 {
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.experimental$log10' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.experimental$log10 %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}

// -----

// ****************** cuda_tile.experimental$log1p ******************

cuda_tile.module @log1p_different_element_type_type {// expected-note @below{{prior use here}}
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.experimental$log1p %arg0 : !cuda_tile.tile<2x4x8xf16>
    }
}

// -----

cuda_tile.module @log1p_different_shape {// expected-note @below{{prior use here}}
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.experimental$log1p %arg0 : !cuda_tile.tile<4x2x8xf32>
    }
}

// -----

cuda_tile.module @log1p_different_rank {// expected-note @below{{prior use here}}
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xf32>) {
        // expected-error @below{{use of value '%arg0' expects different type than prior uses}}
        %0 = cuda_tile.experimental$log1p %arg0 : !cuda_tile.tile<1x2x4x8xf32>
    }
}

// -----

cuda_tile.module @log1p_invalid_type_i32 {
    experimental$func @func(%arg0: !cuda_tile.tile<2x4x8xi32>) {
        // expected-error @below{{'cuda_tile.experimental$log1p' op operand #0 must be tile of f16 or bf16 or f32 or f64 values, but got '!cuda_tile.tile<2x4x8xi32>'}}
        %0 = cuda_tile.experimental$log1p %arg0 : !cuda_tile.tile<2x4x8xi32>
    }
}