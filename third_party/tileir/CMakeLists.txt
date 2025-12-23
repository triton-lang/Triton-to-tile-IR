include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Require modern compilers: GCC/Clang version must be >= 13 (i.e., > 13)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
    message(FATAL_ERROR "C++ compiler version must be >= 13 (found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
  endif()
endif()
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  if(CMAKE_C_COMPILER_VERSION VERSION_LESS 13)
    message(FATAL_ERROR "C compiler version must be >= 13 (found ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")
  endif()
endif()

set(HAVE_CUDA_TILE_INSTALL OFF)
if(DEFINED ENV{CUDA_TILE_INSTALL_DIR})
  set(CUDA_TILE_INSTALL_DIR $ENV{CUDA_TILE_INSTALL_DIR})
  set(CUDA_TILE_INCLUDE_DIRS ${CUDA_TILE_INSTALL_DIR}/include/include)
  set(CUDA_TILE_LIBRARY_PATH ${CUDA_TILE_INSTALL_DIR}/lib)
  set(CUDA_TILE_MLIR_INCLUDE_DIRS $ENV{CUDA_TILE_MLIR_INCLUDE_DIRS})
  set(HAVE_CUDA_TILE_INSTALL ON)
else()
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tileir_src/.git)
    execute_process(
      COMMAND git clone --depth 1 https://github.com/NVIDIA/cuda-tile tileir_src
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE CUDA_TILE_GIT_CLONE_RESULT
    )
    if(NOT CUDA_TILE_GIT_CLONE_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to clone https://github.com/NVIDIA/cuda-tile into ${CMAKE_CURRENT_SOURCE_DIR}/tileir_src")
    endif()
  endif()
  set(CUDA_TILE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tileir_src)

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env LLVM_SYSPATH=${LLVM_SYSPATH} LLVM_EXTERNAL_LIT=${LLVM_SYSPATH}/bin/llvm-lit bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cuda_tile.sh ${CUDA_TILE_SOURCE_DIR}
    WORKING_DIRECTORY ${CUDA_TILE_SOURCE_DIR}
    RESULT_VARIABLE CUDA_TILE_BUILD_RESULT
  )
  if(NOT CUDA_TILE_BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "cuda_tile build failed with code: ${CUDA_TILE_BUILD_RESULT}")
  endif()

  # After successful build, set variables for downstream usage in this configure
  set(CUDA_TILE_INSTALL_DIR ${CUDA_TILE_SOURCE_DIR}/build/install)
  set(CUDA_TILE_LIBRARY_PATH ${CUDA_TILE_INSTALL_DIR}/lib)
  set(CUDA_TILE_INCLUDE_DIRS ${CUDA_TILE_INSTALL_DIR}/include/include)
  set(CUDA_TILE_MLIR_INCLUDE_DIRS ${CUDA_TILE_SOURCE_DIR}/build/include)
  set(HAVE_CUDA_TILE_INSTALL ON)
  # Export to CACHE and ENV for downstream visibility
  set(CUDA_TILE_INSTALL_DIR ${CUDA_TILE_INSTALL_DIR} CACHE PATH "CUDA Tile install dir" FORCE)
  set(CUDA_TILE_INCLUDE_DIRS ${CUDA_TILE_INCLUDE_DIRS} CACHE PATH "CUDA Tile include dirs" FORCE)
  set(CUDA_TILE_LIBRARY_PATH ${CUDA_TILE_LIBRARY_PATH} CACHE PATH "CUDA Tile library path" FORCE)
  set(HAVE_CUDA_TILE_INSTALL ${HAVE_CUDA_TILE_INSTALL} CACHE BOOL "Has cuda tile install" FORCE)
  set(ENV{CUDA_TILE_INSTALL_DIR} ${CUDA_TILE_INSTALL_DIR})
  set(ENV{CUDA_TILE_INCLUDE_DIRS} ${CUDA_TILE_INCLUDE_DIRS})
  set(ENV{CUDA_TILE_LIBRARY_PATH} ${CUDA_TILE_LIBRARY_PATH})
endif()



# Set up tileir specific variables for lit tests
set(TRITON_BINARY_DIR ${CMAKE_BINARY_DIR})
set(TRITON_CUDA_TILE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TRITON_CUDA_TILE_TOOL_DIR ${CMAKE_BINARY_DIR}/third_party/tileir/tools/triton-cuda-tile-opt)
set(TRITON_CUDA_TILE_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
set(TRITON_CUDA_TILE_LIBRARY_DIR ${CMAKE_BINARY_DIR}/third_party/tileir/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

if(HAVE_CUDA_TILE_INSTALL)
  include_directories(${CUDA_TILE_INCLUDE_DIRS})
  include_directories(${CUDA_TILE_MLIR_INCLUDE_DIRS})
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/triton-cuda-tile-opt)

if(TRITON_BUILD_PYTHON_MODULE AND HAVE_CUDA_TILE_INSTALL)
  add_triton_plugin(TritonTileIR ${CMAKE_CURRENT_SOURCE_DIR}/triton_tileir.cc LINK_LIBS TritonToTileIR TritonTileIRTransforms)
  target_link_libraries(TritonTileIR PRIVATE Python3::Module pybind11::headers)
  target_link_libraries(TritonTileIR PRIVATE ${CUDA_TILE_LIBRARY_PATH}/libCudaTileTransforms.a)
  target_link_libraries(TritonTileIR PRIVATE ${CUDA_TILE_LIBRARY_PATH}/libCudaTileBytecodeWriter.a)
  target_link_libraries(TritonTileIR PRIVATE ${CUDA_TILE_LIBRARY_PATH}/libCudaTileBytecodeCommon.a)
elseif(TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "Skip building TritonTileIR Python plugin: CUDA_TILE_INSTALL_DIR not detected")
endif()
