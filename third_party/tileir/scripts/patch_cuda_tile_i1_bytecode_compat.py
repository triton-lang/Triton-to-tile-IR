#!/usr/bin/env python3
"""Patch copied cuda_tile bytecode sources for cross-LLVM i1 compatibility.

The checked-out TileIR sources remain unchanged. This compatibility bridge is
applied only to the clean build copy and can be removed once the bytecode
boundary is stable across the LLVM versions used by Triton and tileiras.
"""

from __future__ import annotations

from pathlib import Path
import sys


def patch_writer(path: Path) -> None:
    text = path.read_text()
    helper_old = """/// Helper function to serialize the APFloat representation of a FloatAttr.
static void writeAPFloatRepresentation(const APFloat &apFloat,
                                       EncodingWriter &writer) {
  writeAPInt(apFloat.bitcastToAPInt(), writer);
}
"""
    helper_new = helper_old + """
/// Serialize i1 dense elements using a stable packed-bits format instead of
/// LLVM's version-dependent raw storage layout.
static void packI1DenseElements(DenseElementsAttr denseAttr,
                                SmallVectorImpl<char> &packedData) {
  assert(denseAttr.getElementType().isInteger(1) &&
         "expected i1 dense elements attribute");
  if (denseAttr.isSplat()) {
    packedData.resize(1);
    packedData[0] = denseAttr.getSplatValue<APInt>().isZero() ? 0x00 : 0xFF;
    return;
  }
  size_t numElements = denseAttr.getNumElements();
  packedData.assign(llvm::divideCeil(numElements, size_t{8}), 0);
  size_t idx = 0;
  for (const APInt &value : denseAttr.getValues<APInt>()) {
    if (!value.isZero())
      packedData[idx / 8] |= static_cast<char>(1u << (idx % 8));
    ++idx;
  }
}
"""
    serialize_old = """  LogicalResult serializeAttribute(Attribute attr, EncodingWriter &writer) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
      // Get the raw data buffer in little-endian format.
      ArrayRef<char> rawData = denseAttr.getRawData();
"""
    serialize_new = """  LogicalResult serializeAttribute(Attribute attr, EncodingWriter &writer) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
      if (denseAttr.getElementType().isInteger(1)) {
        SmallVector<char, 8> packedData;
        packI1DenseElements(denseAttr, packedData);
        writer.writeVarInt(packedData.size());
        writer.write(packedData.data(), packedData.size());
        return success();
      }

      // Get the raw data buffer in little-endian format.
      ArrayRef<char> rawData = denseAttr.getRawData();
"""
    if "packI1DenseElements" not in text:
        if helper_old not in text:
            raise SystemExit("failed to find writer helper anchor")
        text = text.replace(helper_old, helper_new, 1)
    if "SmallVector<char, 8> packedData;" not in text:
        if serialize_old not in text:
            raise SystemExit("failed to find writer serialize anchor")
        text = text.replace(serialize_old, serialize_new, 1)
    path.write_text(text)


def patch_reader(path: Path) -> None:
    text = path.read_text()
    helper_old = """static const uint8_t kTileIRBytecodeMagic[8] = {
    0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00,
};
"""
    helper_new = helper_old + """
static bool shouldUnpackLegacyPackedI1RawData(cuda_tile::TileType tileType,
                                              ArrayRef<char> rawData) {
  if (!tileType.getElementType().isInteger(1))
    return false;
  size_t numElements = tileType.getNumElements();
  size_t packedSize = llvm::divideCeil(numElements, size_t{8});
  return rawData.size() == packedSize && rawData.size() != numElements;
}

static void unpackLegacyPackedI1RawData(ArrayRef<char> rawData,
                                        size_t numElements,
                                        SmallVectorImpl<char> &unpackedData) {
  unpackedData.assign(numElements, 0);
  for (size_t i = 0; i < numElements; ++i) {
    uint8_t byte = static_cast<uint8_t>(rawData[i / 8]);
    unpackedData[i] = (byte & (1u << (i % 8))) ? 1 : 0;
  }
}
"""
    raw_old = """    // Convert ArrayRef<uint8_t> to ArrayRef<char>.
    ArrayRef<char> rawData(reinterpret_cast<const char *>(rawUint8Data.data()),
                           rawUint8Data.size());
    // Validate the buffer size and format.
"""
    raw_new = """    // Convert ArrayRef<uint8_t> to ArrayRef<char>.
    ArrayRef<char> rawData(reinterpret_cast<const char *>(rawUint8Data.data()),
                           rawUint8Data.size());

    // Legacy TileIR bytecode used packed bits for i1. Newer LLVM uses one byte
    // per element, so normalize before calling DenseElementsAttr APIs.
    SmallVector<char, 64> unpackedI1Data;
    if (shouldUnpackLegacyPackedI1RawData(tileType, rawData)) {
      unpackLegacyPackedI1RawData(rawData, tileType.getNumElements(),
                                  unpackedI1Data);
      rawData = unpackedI1Data;
    }

    // Validate the buffer size and format.
"""
    if "shouldUnpackLegacyPackedI1RawData" not in text:
        if helper_old not in text:
            raise SystemExit("failed to find reader helper anchor")
        text = text.replace(helper_old, helper_new, 1)
    if "SmallVector<char, 64> unpackedI1Data;" not in text:
        if raw_old not in text:
            raise SystemExit("failed to find reader raw-data anchor")
        text = text.replace(raw_old, raw_new, 1)
    path.write_text(text)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <cuda_tile_src_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    writer = root / "lib/Bytecode/Writer/BytecodeWriter.cpp"
    reader = root / "lib/Bytecode/Reader/BytecodeReader.cpp"
    if not writer.is_file() or not reader.is_file():
        raise SystemExit(f"missing cuda_tile bytecode sources under {root}")
    patch_writer(writer)
    patch_reader(reader)
    print("[patch] Applied cuda_tile i1 bytecode compatibility patch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
