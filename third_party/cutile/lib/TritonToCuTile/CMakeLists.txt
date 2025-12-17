set(CUDA_TILE_INSTALL_DIR $ENV{CUDA_TILE_INSTALL_DIR})

add_triton_library(TritonToCuTile
    TritonToCudaTilePass.cpp
    Utils.cpp

    DEPENDS
    TritonToCuTileConversionPassIncGen

    LINK_LIBS PUBLIC
    # order is important
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonAnalysis
    TritonIR
    Utils
    ${CUDA_TILE_INSTALL_DIR}/lib/libCudaTileDialect.a
    ${CUDA_TILE_INSTALL_DIR}/lib/libCudaTileTransforms.a
)
