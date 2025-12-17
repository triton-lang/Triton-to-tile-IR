// RUN: %round_trip_test %s %t


cuda_tile.module @kernels {
  // Function with a very long name to test string table handling
  cuda_tile.experimental$func @this_is_an_extremely_long_function_name_that_will_test_the_string_table_handling_in_the_bytecode_reader_and_help_to_increase_coverage_by_testing_unusual_cases_and_boundary_conditions_in_the_string_section_parser(%arg0: i32) -> i32 {
    cuda_tile.return %arg0 : i32
  }
  
  // Function with many parameters to test function signature parsing
  cuda_tile.experimental$func @many_parameters_function(
    %p0: i32, %p1: i32, %p2: i32, %p3: i32, %p4: i32, 
    %p5: i32, %p6: i32, %p7: i32, %p8: i32, %p9: i32,
    %p10: i32, %p11: i32, %p12: i32, %p13: i32, %p14: i32,
    %p15: i32, %p16: i32, %p17: i32, %p18: i32, %p19: i32
  ) -> i32 {
    cuda_tile.return %p0 : i32
  }
  
  // Function with many results to test result type handling
  cuda_tile.experimental$func @many_results_function(%arg0: i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
    cuda_tile.return %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
  }
}
