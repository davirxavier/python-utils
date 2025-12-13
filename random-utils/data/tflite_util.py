from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def make_representative_gen(test_ds, max_samples=100):
    def rep_gen():
        for img, _ in test_ds.batch(1).repeat().take(max_samples):
            yield [img]

    return rep_gen


def quantize_model(model_or_path,
                   output_path="model_quant.tflite",
                   representative_ds=None):
    """
    Quantizes a Keras model to TFLite.

    Args:
        model_or_path: Either:
            - a Keras model object already loaded in memory, OR
            - a string/path pointing to a .h5 file
        output_path: Where to save the .tflite file
        representative_ds: Required for full INT8
        int8_full: Whether to force full INT8 quantization

    Returns:
        Path to the saved .tflite model
    """

    if isinstance(model_or_path, (str, Path)):
        print(f"Loading model from file: {model_or_path}")
        model = tf.keras.models.load_model(model_or_path, compile=False)
    else:
        model = model_or_path

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if representative_ds is None:
        raise ValueError("INT8 full quantization requires a representative dataset.")

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_gen(representative_ds)

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)

    print(f"Saved quantized model to: {output_path}")
    return output_path


def print_confusion_matrix(y_true, y_pred, class_names):
    """
    Prints a nicely formatted confusion matrix in the terminal.
    """
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)

    # Header
    header = " " * 12 + "".join([f"{name:<10}" for name in class_names])
    print(header)
    print("-" * (12 + 10 * n))

    # Rows
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<12}" + "".join([f"{val:<10}" for val in row])
        print(row_str)


def evaluate_on_test_tflite(tflite_model_path, test_ds, class_names=None):
    """
    Evaluates a TFLite model (float or quantized) on the test dataset.
    Handles float32, uint8, and int8 quantization for both input and output.

    Args:
        tflite_model_path: Path to the .tflite model file.
        test_ds: tf.data.Dataset yielding (images, one-hot labels).
        class_names: Optional list of names for each class.
    """
    print("\n=== Running evaluation on test dataset (TFLite) ===")

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index  = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_dtype  = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    in_scale, in_zero_point   = input_details[0]["quantization"]
    out_scale, out_zero_point = output_details[0]["quantization"]

    quantized_input  = in_scale  != 0  # True if uint8/int8 input
    quantized_output = out_scale != 0  # True if uint8/int8 output

    print(f"- Input dtype:  {input_dtype}, quantized={quantized_input}")
    print(f"- Output dtype: {output_dtype}, quantized={quantized_output}")

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        images = images.numpy()  # float32 [0,1]
        labels = labels.numpy()

        for i in range(images.shape[0]):
            img = images[i:i+1]

            # ---- Handle input quantization ----
            if quantized_input:
                # Convert float32 [0,1] to quantized domain
                img_q = img / in_scale + in_zero_point
                img_q = np.clip(img_q, 0, 255).astype(input_dtype)
                interpreter.set_tensor(input_index, img_q)
            else:
                interpreter.set_tensor(input_index, img.astype(input_dtype))

            # Run model
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)[0]

            # ---- Handle output dequantization ----
            if quantized_output:
                output = (output.astype(np.float32) - out_zero_point) * out_scale

            y_true.append(np.argmax(labels[i]))
            y_pred.append(np.argmax(output))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Total accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\nTotal samples: {len(y_true)}")
    print(f"Correct predictions: {(accuracy/100)*len(y_true):.0f} ({accuracy:.2f}%)\n")

    # Class names
    if class_names is None:
        num_classes = len(np.unique(y_true))
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Detailed per-class report
    print("Per-class metrics (Precision, Recall, F1-score):")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    print("Confusion matrix:")
    print_confusion_matrix(y_true, y_pred, class_names)

    print("\nTFLite test evaluation complete.\n")

