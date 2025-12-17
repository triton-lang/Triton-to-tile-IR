add_subdirectory(Bytecode)
# Skipping CAPI if CUDA_TILE_ENABLE_CAPI is OFF.
if(CUDA_TILE_ENABLE_CAPI)
  add_subdirectory(CAPI)
endif()
add_subdirectory(Dialect)
