from nebula3.common.ttypes import PropertyType


# TODO: support more types, after vector type is supported,
#  we can use the length of the vector to get the feature dim
def get_feature_dim(nebula_col):
    """
    Determines the feature dimension based on the property type definition
    from NebulaGraph schema.

    This function maps NebulaGraph's column type to a feature dimension
    expected by PyG (usually 1 for scalar types).

    Args:
        nebula_col: A column type object from NebulaGraph schema,
                    typically from 'desc tag' or 'desc edge'.

    Returns:
        int: The dimension of the feature associated with the column.

    Raises:
        ValueError: If the property type is not supported or unknown.
    """
    type_code = nebula_col.type.type
    type_name = PropertyType._VALUES_TO_NAMES.get(type_code, "UNKNOWN")

    if type_name == "UNKNOWN":
        raise ValueError(f"Unsupported type: {type_code}")

    if type_name == "FIXED_STRING":
        return nebula_col.type.type_length
    elif type_name == "STRING":
        return 1
    elif type_name in ("INT", "INT64", "INT32", "INT16", "INT8"):
        return 1
    elif type_name in ("FLOAT", "DOUBLE"):
        return 1
    elif type_name == "BOOL":
        return 1
    elif type_name in ("DATE", "TIME", "DATETIME", "TIMESTAMP"):
        return 1
    elif type_name == "GEOGRAPHY":
        return 1
    else:
        raise ValueError(f"Unsupported type: {type_name}")
