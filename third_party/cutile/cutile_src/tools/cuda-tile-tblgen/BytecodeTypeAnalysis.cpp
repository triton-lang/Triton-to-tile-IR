#include "BytecodeTypeAnalysis.h"

#include "llvm/TableGen/Error.h"


using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// BytecodeTypeParameter Implementation.
//===----------------------------------------------------------------------===//

BytecodeTypeParameter::Kind
BytecodeTypeParameter::classifyParameter(const AttrOrTypeParameter &param) {
  StringRef cppType = param.getCppType();
  StringRef cppStorageType = param.getCppStorageType();

  // ArrayRefParameter sets cppStorageType to SmallVector.
  if (cppStorageType.contains("SmallVector<int64_t>"))
    return Kind::Int64Array;
  if (cppStorageType.contains("SmallVector<int32_t>"))
    return Kind::Int32Array;
  if (cppStorageType.contains("DenseI32ArrayAttr"))
    return Kind::DenseI32Array;

  // Check for Type parameters.
  if (cppType.contains("Type")) {
    if (cppType.contains("cuda_tile::"))
      return Kind::SpecificType;
    return Kind::GenericType;
  }

  // Check for optional enum attributes.
  if (param.isOptional() && cppType.contains("Attr"))
    return Kind::OptionalEnum;

  // Unsupported parameter type.
  PrintFatalError("Unsupported parameter type for bytecode generation: " +
                  cppType.str() + " (storage: " + cppStorageType.str() + ")");
}

BytecodeTypeParameter::BytecodeTypeParameter(const AttrOrTypeParameter &param)
    : name(param.getName().str()), accessorName(param.getAccessorName()),
      cppType(param.getCppType().str()),
      cppStorageType(param.getCppStorageType().str()),
      isOptional(param.isOptional()), kind(classifyParameter(param)) {

  // Extract version from wrapped parameters (CudaTileTypeParam adds sinceVersion).
  // Use getValue() to check if field exists before accessing - raw parameters
  // like ArrayRefParameter don't have this field in their class hierarchy.
  if (const auto *defInit = dyn_cast_if_present<DefInit>(param.getDef())) {
    const Record *paramRecord = defInit->getDef();
    if (paramRecord->getValue("sinceVersion") &&
        !paramRecord->isValueUnset("sinceVersion"))
      sinceVersion = paramRecord->getValueAsString("sinceVersion").str();
  }

  // Extract default value if present.
  if (auto defValue = param.getDefaultValue())
    defaultValue = defValue->str();

  // Extract enum type name for OptionalEnum kind.
  if (kind == Kind::OptionalEnum) {
    auto split = StringRef(cppType).rsplit("::");
    if (split.second.empty() || !split.second.ends_with("Attr"))
      PrintFatalError("OptionalEnum parameter type must end with 'Attr': " +
                      cppType);
    enumTypeName = split.second.drop_back(4).str();
  }

  // Compute if this parameter is optional with null default (Type, Enum, etc.).
  usesOptionalTypeFlags = isOptional && (defaultValue == cppType + "()");
}

//===----------------------------------------------------------------------===//
// CudaTileType Implementation.
//===----------------------------------------------------------------------===//

CudaTileType::CudaTileType(const AttrOrTypeDef &typeDef, unsigned tagValue,
                           StringRef version)
    : typeName(typeDef.getCppClassName().str()),
      qualifiedTypeName(typeDef.getDialect().getCppNamespace().str() +
                        "::" + typeDef.getCppClassName().str()),
      typeTagValue(tagValue), sinceVersion(version.str()),
      needsReverseOrder(typeDef.getCppClassName() == "TileType") {


  // Analyze and validate all parameters.
  for (const auto &attrParam : typeDef.getParameters()) {
    BytecodeTypeParameter param(attrParam);

    // Validate: All parameters must have version information.
    if (!skipVersionCheck && param.sinceVersion.empty())
      PrintFatalError(
          "Parameter '" + param.name + "' in type '" + typeName +
          "' must be wrapped with CudaTileTypeParam or "
          "CudaTileConstrainedTypeParam to have version information.");

    // Detect optional Type parameters with null defaults for flag-based.
    if (param.usesOptionalTypeFlags) {
      hasOptionalTypeParams = true;
      if (!skipVersionCheck &&
          (firstOptionalTypeParamVersion.empty() ||
           param.sinceVersion < firstOptionalTypeParamVersion))
        firstOptionalTypeParamVersion = param.sinceVersion;
    }

    // Validate: Non-optional parameters introduced after type need defaults.
    if (!skipVersionCheck && !param.isOptional &&
        param.sinceVersion != sinceVersion) {
      if (param.defaultValue.empty())
        PrintFatalError(
            "Parameter '" + param.name + "' in type '" + typeName +
            "' was introduced in version " + param.sinceVersion +
            " after the type (version " + sinceVersion +
            "). It must have a default value for backward compatibility.");
    }

    parameters.push_back(std::move(param));
  }
}

//===----------------------------------------------------------------------===//
// BuiltinType Implementation.
//===----------------------------------------------------------------------===//

BuiltinType::BuiltinType(StringRef name, StringRef qualifiedType, unsigned tag,
                         StringRef version, unsigned width, StringRef floatType)
    : enumName(name.str()), qualifiedTypeName(qualifiedType.str()),
      typeTagValue(tag), sinceVersion(version.str()), integerBitWidth(width),
      floatMlirTypeName(floatType.str()) {}

//===----------------------------------------------------------------------===//
// Analysis Entry Point.
//===----------------------------------------------------------------------===//

FailureOr<BytecodeTypeStructure>
mlir::tblgen::analyzeBytecodeTypes(const RecordKeeper &records) {
  BytecodeTypeStructure structure;

  // Build map of CudaTileTypeDef for matching.
  StringMap<const Record *> cudaTileTypeDefRecords;
  for (const Record *typeRecord :
       records.getAllDerivedDefinitions("CudaTileTypeDef"))
    cudaTileTypeDefRecords[AttrOrTypeDef(typeRecord).getCppClassName()] =
        typeRecord;

  // Build map of builtin type versions, stripping "CudaTile_" prefix.
  // Maps enum name (e.g., "Int32") to version string for efficient lookup.
  StringMap<StringRef> builtinTypeVersions;
  for (const Record *aliasRecord :
       records.getAllDerivedDefinitions("CudaTileTypeAlias")) {
    StringRef aliasName = aliasRecord->getName();
    StringRef version = aliasRecord->getValueAsString("sinceVersion");
    if (aliasName.starts_with("CudaTile_"))
      builtinTypeVersions[aliasName.drop_front(9)] = version;
  }

  // Helper to lookup version from tag enum name.
  auto lookupBuiltinVersion = [&](StringRef enumName) -> StringRef {
    StringRef version = builtinTypeVersions.lookup(enumName);
    if (version.empty())
      PrintFatalError("No version found for builtin type: " + enumName.str());
    return version;
  };

  // Process all BytecodeTypeTag records.
  for (const Record *record :
       records.getAllDerivedDefinitions("BytecodeTypeTag")) {
    StringRef enumName = record->getValueAsString("cppTypeName");
    unsigned tagValue = record->getValueAsInt("typeTagValue");

    // Add to enum.
    structure.allTypeTags.emplace_back(enumName.str(), tagValue);

    // Categorize and process based on subclass.
    if (record->isSubClassOf("IntegerTypeTag")) {
      structure.builtinSerializableTypes.emplace_back(
          enumName, "IntegerType", tagValue, lookupBuiltinVersion(enumName),
          record->getValueAsInt("integerBitWidth"));
    } else if (record->isSubClassOf("FloatTypeTag")) {
      structure.builtinSerializableTypes.emplace_back(
          enumName, "FloatType", tagValue, lookupBuiltinVersion(enumName), 0,
          record->getValueAsString("floatMlirTypeName"));
    } else if (record->isSubClassOf("CudaTileTypeTag")) {
      auto it = cudaTileTypeDefRecords.find(enumName);
      if (it != cudaTileTypeDefRecords.end()) {
        AttrOrTypeDef typeDef(it->second);
        StringRef typeVersion =
            typeDef.getDef()->getValueAsString("sinceVersion");
        structure.cudaTileTypes.emplace_back(typeDef, tagValue, typeVersion);
      }
    }
  }

  return structure;
}
