set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonToCuTile)
add_public_tablegen_target(TritonToCuTileConversionPassIncGen)
