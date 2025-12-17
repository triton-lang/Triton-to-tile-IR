// Test 250.1 features: operands, results, and attributes.

// RUN: cuda-tile-translate -test-cudatile-roundtrip -no-implicit-module -bytecode-version=250.1 %s | FileCheck %s

cuda_tile.module @version_250_1_features {
  // Test case 1: Operand parsing - validates 250.1 optional operand are correctly parsed.
  entry @test_operand_parsing() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token_in = make_token : !cuda_tile.token
    %token_out = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) 
      token = %token_in : !cuda_tile.token -> !cuda_tile.token
    // CHECK: %{{.*}} = testing$bytecode_test_evolution(%{{.*}} : !cuda_tile.tile<2xf32>) token = %{{.*}} : token -> token
    return
  }

  // Test case 2: Result parsing - validates 250.1 results are correctly parsed and usable.
  entry @test_result_parsing() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token1 = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    // CHECK: %[[TOKEN1:.*]] = testing$bytecode_test_evolution(%{{.*}} : !cuda_tile.tile<2xf32>) -> token
    %token2 = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    // CHECK: %[[TOKEN2:.*]] = testing$bytecode_test_evolution(%{{.*}} : !cuda_tile.tile<2xf32>) -> token
    // Use parsed results to validate correct type preservation during deserialization
    %joined_tokens = join_tokens %token1, %token2 : !cuda_tile.token
    // CHECK: %{{.*}} = join_tokens %[[TOKEN1]], %[[TOKEN2]] : token
    return
  }

  // Test case 3: Attribute parsing - validates 250.1 non-default attributes are correctly parsed.
  entry @test_attribute_parsing() {
    testing$bytecode_test_new_attribute new_flag new_param = 123
    // CHECK: bytecode_test_new_attribute new_flag new_param = 123
    return
  }

  // Test case 4: Type parameter with DefaultValuedParameter from 250.1.
  entry @test_type_explicit_default(
      %arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f16>) {
    // CHECK: testing$bytecode_test_evolved<f32, f16>
    return
  }
  
  // Test case 5: OptionalParameter<Attr> with value.
  entry @test_optional_attr(%arg0: !cuda_tile.testing$bytecode_test_evolved<f64, i32, zero>) {
    // CHECK: testing$bytecode_test_evolved<f64, zero>
    return
  }
  
  // Test case 6: OptionalParameter<Type> with value.
  entry @test_optional_type_present(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, i32, zero, f16>) {
    // CHECK: testing$bytecode_test_evolved<f32, zero, f16>
    return
  }
  
  // Test case 7: OptionalParameter<Type> NULL.
  entry @test_optional_type_null(%arg0: !cuda_tile.testing$bytecode_test_evolved<f64, i32>) {
    // CHECK: testing$bytecode_test_evolved<f64>
    return
  }
  
  // Test case 8: All defaults/nulls.
  entry @test_all_defaults(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32>) {
    // CHECK: testing$bytecode_test_evolved<f32>
    return
  }
}
