import math
import os
import tensorflow as tf

# Map TFLite ops to TFLite Micro resolver functions
TFLM_OP_RESOLVER_MAP = {
    # Convolution / Dense
    "CONV_2D": "AddConv2D",
    "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D",
    "FULLY_CONNECTED": "AddFullyConnected",
    "TRANSPOSE_CONV": "AddTransposeConv",

    # Pooling
    "MAX_POOL_2D": "AddMaxPool2D",
    "AVERAGE_POOL_2D": "AddAveragePool2D",
    "L2_POOL_2D": "AddL2Pool2D",

    # Activations
    "RELU": "AddRelu",
    "RELU6": "AddRelu6",
    "LEAKY_RELU": "AddLeakyRelu",
    "PRELU": "AddPrelu",
    "LOGISTIC": "AddLogistic",
    "SOFTMAX": "AddSoftmax",
    "TANH": "AddTanh",
    "ELU": "AddElu",

    # Elementwise ops
    "ADD": "AddAdd",
    "SUB": "AddSub",
    "MUL": "AddMul",
    "DIV": "AddDiv",
    "MAXIMUM": "AddMaximum",
    "MINIMUM": "AddMinimum",
    "SQUARED_DIFFERENCE": "AddSquaredDifference",

    # Reduction
    "MEAN": "AddMean",
    "SUM": "AddSum",
    "REDUCE_MAX": "AddReduceMax",
    "REDUCE_MIN": "AddReduceMin",
    "REDUCE_PROD": "AddReduceProd",

    # Shape / Tensor manipulation
    "RESHAPE": "AddReshape",
    "TRANSPOSE": "AddTranspose",
    "SQUEEZE": "AddSqueeze",
    "EXPAND_DIMS": "AddExpandDims",
    "PACK": "AddPack",
    "UNPACK": "AddUnpack",
    "CONCATENATION": "AddConcatenation",
    "SPLIT": "AddSplit",
    "SPLIT_V": "AddSplitV",
    "STRIDED_SLICE": "AddStridedSlice",
    "SLICE": "AddSlice",
    "GATHER": "AddGather",
    "GATHER_ND": "AddGatherNd",

    # Padding / Resize
    "PAD": "AddPad",
    "PADV2": "AddPadV2",
    "RESIZE_NEAREST_NEIGHBOR": "AddResizeNearestNeighbor",
    "RESIZE_BILINEAR": "AddResizeBilinear",

    # Quantization
    "QUANTIZE": "AddQuantize",
    "DEQUANTIZE": "AddDequantize",

    # Math
    "ABS": "AddAbs",
    "NEG": "AddNeg",
    "LOG": "AddLog",
    "EXP": "AddExp",
    "SQRT": "AddSqrt",
    "RSQRT": "AddRsqrt",
    "POW": "AddPow",
    "SIN": "AddSin",
    "COS": "AddCos",

    # Comparison / logic
    "EQUAL": "AddEqual",
    "NOT_EQUAL": "AddNotEqual",
    "GREATER": "AddGreater",
    "GREATER_EQUAL": "AddGreaterEqual",
    "LESS": "AddLess",
    "LESS_EQUAL": "AddLessEqual",
    "LOGICAL_AND": "AddLogicalAnd",
    "LOGICAL_OR": "AddLogicalOr",
    "LOGICAL_NOT": "AddLogicalNot",

    # Other commonly used ops
    "ARG_MAX": "AddArgMax",
    "ARG_MIN": "AddArgMin",
    "SELECT": "AddSelect",
    "SELECT_V2": "AddSelectV2",
    "WHERE": "AddWhere",
}


def generate_model_header(model_path, header_output_path, prefix):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    model_name = prefix
    macro_prefix = model_name.upper()
    namespace_name = model_name
    include_guard = f"{macro_prefix}_H_"

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Input shape
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]["shape"]
    _, input_height, input_width, input_channels = input_shape

    # Output shape (flattened size)
    output_details = interpreter.get_output_details()
    output_shape = output_details[0]["shape"]
    output_tensor_size = 1
    for d in output_shape:
        output_tensor_size *= d

    # Extract operator types (correct way)
    ops_details = interpreter._get_ops_details()
    distinct_ops = sorted(set(op["op_name"] for op in ops_details))
    num_distinct_ops = len(distinct_ops)

    # Read model binary
    with open(model_path, "rb") as f:
        model_data = f.read()

    model_size = len(model_data)

    with open(header_output_path, "w") as f:
        # Include guard
        f.write(f"#ifndef {include_guard}\n")
        f.write(f"#define {include_guard}\n\n")

        # Includes
        f.write("#include <cstdint>\n")
        f.write("#include \"tensorflow/lite/micro/micro_mutable_op_resolver.h\"\n\n")

        # Macros
        f.write(f"#define {macro_prefix}_MODEL_SIZE {model_size}\n")
        f.write(f"#define {macro_prefix}_INPUT_WIDTH {input_width}\n")
        f.write(f"#define {macro_prefix}_INPUT_HEIGHT {input_height}\n")
        f.write(f"#define {macro_prefix}_INPUT_CHANNELS {input_channels}\n")
        f.write(f"#define {macro_prefix}_OUTPUT_TENSOR_SIZE {output_tensor_size}\n")
        f.write(f"#define {macro_prefix}_DISTINCT_OPS_COUNT {num_distinct_ops}\n\n")

        # Namespace begin
        f.write(f"namespace {namespace_name} {{\n\n")

        # Model data
        f.write("alignas(16) const unsigned char tflite[] = {\n")
        for i in range(0, len(model_data), 12):
            chunk = model_data[i:i + 12]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n\n")
        f.write(f"const unsigned int tflite_len = {model_size};\n\n")

        # Resolver function
        f.write("// Auto-generated TFLite Micro operator registration\n")
        f.write("inline void RegisterOps(\n")
        f.write(
            f"    tflite::MicroMutableOpResolver<{macro_prefix}_DISTINCT_OPS_COUNT>& resolver) {{\n"
        )

        for op in distinct_ops:
            resolver_fn = TFLM_OP_RESOLVER_MAP.get(op)
            if resolver_fn:
                f.write(f"  resolver.{resolver_fn}();\n")
            else:
                f.write(f"  // ERROR: Unsupported op '{op}'\n")

        f.write("}\n\n")

        # Namespace end
        f.write(f"}}  // namespace {namespace_name}\n\n")

        # End include guard
        f.write(f"#endif  // {include_guard}\n")

    print("Model header generated successfully:")
    print(f"  Output file: {header_output_path}")
    print(f"  Model size: {model_size} bytes")
    print(f"  Distinct ops ({num_distinct_ops}):")
    for op in distinct_ops:
        print(f"   - {op}")


if __name__ == "__main__":
    model_path = "detection_quantized.tflite"
    header_output_path = (
        "/home/xav/Documents/git/arduinoprojects/"
        "myprojects/esp-cam-cat-detector/src/general/model_data.h"
    )
    prefix = "model_data"

    generate_model_header(model_path, header_output_path, prefix)
