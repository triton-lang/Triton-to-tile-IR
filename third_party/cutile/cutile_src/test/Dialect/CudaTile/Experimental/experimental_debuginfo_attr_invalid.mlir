// RUN: not cuda-tile-opt --split-input-file --mlir-print-debuginfo --allow-unregistered-dialect %s 2>&1 | FileCheck %s
// RUN: not cuda-tile-translate --test-cudatile-roundtrip --no-implicit-module --split-input-file --mlir-print-debuginfo --allow-unregistered-dialect %s 2>&1 | FileCheck %s

// NOTE: This test generates invalid debug info. The presence of invalid debug
// info means that the typical --verify-diagnostics flow used for invalid tests
// will not work for this test as that flow relies on valid debug info. The
// inability to use the --verify-diagnostics flow means that this test is
// expected to fail. The expected failure means that the bytecode
// round_trip_test.sh script will also not work for this test.

// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#di_loc_func = #cuda_tile.di_loc<loc("/tmp/foo.py":7:8) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#di_loc_invalid = #cuda_tile.di_loc<loc("/tmp/foo.py":15:16) in #invalid>
#unknown = loc(unknown)
// end common test setup

// Rule 1: If a function has scope, it must have subprogram scope.
// Test A: Using func
// CHECK: invalid function debug info scope
// CHECK: Function location must have cuda_tile.di_subprogram debug info scope
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(#di_loc_func)
  } loc(#di_loc_block)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#di_loc_func = #cuda_tile.di_loc<loc("/tmp/foo.py":7:8) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#di_loc_invalid = #cuda_tile.di_loc<loc("/tmp/foo.py":15:16) in #invalid>
#unknown = loc(unknown)
// end common test setup

// Rule 2: If a function has subprogram scope, the function name must match the subprogram scope linkage name.
// Test A: Using func
// CHECK: invalid function debug info scope
// CHECK: Function name "foo" does not match subprogram scope linkage name "test"
cuda_tile.module @kernels {
  experimental$func @foo() {
    return loc(#di_loc_func)
  } loc(#di_loc_func)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#di_loc_func = #cuda_tile.di_loc<loc("/tmp/foo.py":7:8) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#di_loc_invalid = #cuda_tile.di_loc<loc("/tmp/foo.py":15:16) in #invalid>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test A: Using func
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(#di_loc_func)
  } loc(#unknown)
}

// -----

// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#di_loc_func = #cuda_tile.di_loc<loc("/tmp/foo.py":7:8) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#di_loc_invalid = #cuda_tile.di_loc<loc("/tmp/foo.py":15:16) in #invalid>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test D: Using func and block scope
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(#di_loc_inner_block)
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#di_loc_func = #cuda_tile.di_loc<loc("/tmp/foo.py":7:8) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#di_loc_invalid = #cuda_tile.di_loc<loc("/tmp/foo.py":15:16) in #invalid>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test E: Using func with operation inside if-else having scope
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    cuda_tile.if %cond {
      cuda_tile.yield loc(#di_loc_func)
    } else {
      cuda_tile.yield
    }
    return
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#di_loc_func = #cuda_tile.di_loc<loc("/tmp/foo.py":7:8) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#di_loc_invalid = #cuda_tile.di_loc<loc("/tmp/foo.py":15:16) in #invalid>
#unknown = loc(unknown)
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test A: Using func + subprogram scope
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(#di_loc_func)
  } loc(#di_loc_invalid)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#di_loc_func = #cuda_tile.di_loc<loc("/tmp/foo.py":7:8) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#di_loc_invalid = #cuda_tile.di_loc<loc("/tmp/foo.py":15:16) in #invalid>
#unknown = loc(unknown)
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test C: Using func + block scope
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(#di_loc_block)
  } loc(#di_loc_invalid)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#di_loc_func = #cuda_tile.di_loc<loc("/tmp/foo.py":7:8) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#di_loc_invalid = #cuda_tile.di_loc<loc("/tmp/foo.py":15:16) in #invalid>
#unknown = loc(unknown)
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test E: Using func + inner block scope
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(#di_loc_inner_block)
  } loc(#di_loc_invalid)
}
