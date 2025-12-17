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
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
// end common test setup

// Rule 1: If a function has scope, it must have subprogram scope.
// Test A: Using func with NameLoc wrapper
// CHECK: invalid function debug info scope
// CHECK: Function location must have cuda_tile.di_subprogram debug info scope
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(#di_loc_func)
  } loc("func_loc"(#di_loc_block))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
// end common test setup

// Rule 1: If a function has scope, it must have subprogram scope.
// Test B: Using func with FusedLoc wrapper  
// CHECK: invalid function debug info scope
// CHECK: Function location must have cuda_tile.di_subprogram debug info scope
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(#di_loc_func)
  } loc(fused[#loc_func, #di_loc_block])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
// end common test setup

// Rule 2: If a function has subprogram scope, the function name must match the subprogram scope linkage name.
// Test A: Using func with NameLoc wrapper
// CHECK: invalid function debug info scope
// CHECK: Function name "foo" does not match subprogram scope linkage name "test"
cuda_tile.module @kernels {
  experimental$func @foo() {
    return loc(#di_loc_func)
  } loc("func_loc"(#di_loc_func))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
// end common test setup

// Rule 2: If a function has subprogram scope, the function name must match the subprogram scope linkage name.
// Test B: Using func with FusedLoc wrapper
// CHECK: invalid function debug info scope
// CHECK: Function name "foo" does not match subprogram scope linkage name "test"
cuda_tile.module @kernels {
  experimental$func @foo() {
    return loc(#di_loc_func)
  } loc(fused[#loc_func, #di_loc_func])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test A: Using func with operation having NameLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc("op_loc"(#di_loc_func))
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test B: Using func with operation having FusedLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(fused[#loc_func, #di_loc_func])
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test C: Using func with operation having CallSiteLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(callsite(#loc_func at #di_loc_func))
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test J: Using func with inner block scope operation having NameLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc("op_loc"(#di_loc_inner_block))
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test K: Using func with inner block scope operation having FusedLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(fused[#loc_func, #di_loc_inner_block])
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test L: Using func with inner block scope operation having CallSiteLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(callsite(#loc_func at #di_loc_inner_block))
  } loc(#unknown)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test M: Using func with if-else operation having NameLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    cuda_tile.if %cond {
      cuda_tile.yield loc("op_loc"(#di_loc_func))
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
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test N: Using func with if-else operation having FusedLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    cuda_tile.if %cond {
      cuda_tile.yield loc(fused[#loc_func, #di_loc_func])
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
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#unknown = loc(unknown)
// end common test setup

// Rule 3: If a function does not have scope, its operations must not have scope.
// Test O: Using func with if-else operation having CallSiteLoc wrapper
// CHECK: invalid operation debug info scope
// CHECK: Operation has debug info scope, but function debug info scope is undefined
cuda_tile.module @kernels {
  experimental$func @test() {
    %cond = cuda_tile.constant <i1: true> : !cuda_tile.tile<i1>
    cuda_tile.if %cond {
      cuda_tile.yield loc(callsite(#loc_func at #di_loc_func))
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
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// TEST A: named location op
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc("di_loc_func"(#di_loc_func))
  } loc(#di_loc_invalid)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// TEST B: named location func
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(#di_loc_func)
  } loc("invalid"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// TEST C: fused location op
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(fused[#loc_func, #di_loc_func])
  } loc(#di_loc_invalid)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// TEST D: fused location func
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(#di_loc_func)
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// TEST D: callsite location op
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(callsite(#loc_func at #di_loc_func))
  } loc(#di_loc_invalid)
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test A1: func + subprogram scope (function NameLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc("op_loc"(#di_loc_func))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test A2: func + subprogram scope (function NameLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(fused[#loc_func, #di_loc_func])
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test A3: func + subprogram scope (function NameLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(callsite(#loc_func at #di_loc_func))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test A4: func + subprogram scope (function FusedLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc("op_loc"(#di_loc_func))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test A5: func + subprogram scope (function FusedLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(fused[#loc_func, #di_loc_func])
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test A6: func + subprogram scope (function FusedLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(callsite(#loc_func at #di_loc_func))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test C1: func + block scope (function NameLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc("op_loc"(#di_loc_block))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test C2: func + block scope (function NameLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(fused[#loc_func, #di_loc_block])
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test C3: func + block scope (function NameLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(callsite(#loc_func at #di_loc_block))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test C4: func + block scope (function FusedLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc("op_loc"(#di_loc_block))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test C5: func + block scope (function FusedLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(fused[#loc_func, #di_loc_block])
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_block = #cuda_tile.di_loc<loc("/tmp/foo.py":9:10) in #block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test C6: func + block scope (function FusedLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(callsite(#loc_func at #di_loc_block))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test E1: func + inner block scope (function NameLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc("op_loc"(#di_loc_inner_block))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test E2: func + inner block scope (function NameLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(fused[#loc_func, #di_loc_inner_block])
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test E3: func + inner block scope (function NameLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(callsite(#loc_func at #di_loc_inner_block))
  } loc("func_loc"(#di_loc_invalid))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test E4: func + inner block scope (function FusedLoc + operation NameLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc("op_loc"(#di_loc_inner_block))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test E5: func + inner block scope (function FusedLoc + operation FusedLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(fused[#loc_func, #di_loc_inner_block])
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#block = #cuda_tile.di_lexical_block<scope = #subprogram, file = #file, line = 3, column = 4>
#inner_block = #cuda_tile.di_lexical_block<scope = #block, file = #file, line = 5, column = 6>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_inner_block = #cuda_tile.di_loc<loc("/tmp/foo.py":11:12) in #inner_block>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":15:16)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 4: Operation scope must match function scope.
// Test E6: func + inner block scope (function FusedLoc + operation CallSiteLoc)
// CHECK: invalid operation debug info scope
// CHECK: Operation debug info scope does not match function debug info scope
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(callsite(#loc_func at #di_loc_inner_block))
  } loc(fused[#loc_invalid, #di_loc_invalid])
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#loc_func = loc("/tmp/foo.py":7:8)
#di_loc_func = #cuda_tile.di_loc<loc(#loc_func) in #subprogram>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// Rule 6: Function location must not be a CallSiteLoc.
// CHECK: invalid function debug info location
// CHECK: Function location must not be a CallSiteLoc
cuda_tile.module @kernels {
  experimental$func @invalid() {
    return loc(#di_loc_func)
  } loc(callsite(#loc_func at #di_loc_func))
}

// -----
// common test setup
#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram1 = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test", compileUnit = #compile_unit, scopeLine = 2>
#subprogram2 = #cuda_tile.di_subprogram<file = #file, line = 1, name = "test", linkageName = "test2", compileUnit = #compile_unit, scopeLine = 2>
#loc_func1 = loc("/tmp/foo.py":7:8)
#loc_func2 = loc("/tmp/foo.py":8:9)
#di_loc_func1 = #cuda_tile.di_loc<loc(#loc_func1) in #subprogram1>
#di_loc_func2 = #cuda_tile.di_loc<loc(#loc_func2) in #subprogram2>
#invalid = #cuda_tile.di_subprogram<file = #file, line = 13, name = "invalid", linkageName = "invalid", compileUnit = #compile_unit, scopeLine = 14>
#loc_invalid = loc("/tmp/foo.py":9:10)
#di_loc_invalid = #cuda_tile.di_loc<loc(#loc_invalid) in #invalid>
// end common test setup

// TODO: TILE-1036
// XCHECK: invalid function debug info location
// XCHECK: Function location must not be a CallSiteLoc
cuda_tile.module @kernels {
  experimental$func @test() {
    return loc(#di_loc_func1)
  } loc(fused[#di_loc_invalid, #di_loc_func1])
}