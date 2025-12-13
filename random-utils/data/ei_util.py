import os
from pathlib import Path

import edgeimpulse as ei
import requests

ei.API_KEY = os.environ["EI_API_KEY"]
ei_weights_urls = {
    "1": {
        "96": {
            "0.05": "transfer-learning-weights/edgeimpulse/MobileNetV2.0_05.96x96.grayscale.bsize_64.lr_0_05.epoch_334.val_loss_4.53.hdf5",
            "0.1": "transfer-learning-weights/edgeimpulse/MobileNetV2.0_1.96x96.grayscale.bsize_64.lr_0_05.epoch_441.val_loss_4.13.val_accuracy_0.2.hdf5",
            "0.35": "transfer-learning-weights/edgeimpulse/MobileNetV2.0_35.96x96.grayscale.bsize_64.lr_0_005.epoch_260.val_loss_3.10.val_accuracy_0.35.hdf5"
        },
    },
    "3": {
        "96": {
            "0.05": "transfer-learning-weights/edgeimpulse/MobileNetV2.0_05.96x96.color.bsize_64.lr_0_05.epoch_574.val_loss_4.22.hdf5",
            "0.1": "transfer-learning-weights/edgeimpulse/MobileNetV2.0_1.96x96.color.bsize_64.lr_0_05.epoch_498.val_loss_3.85.hdf5",
            "0.35": "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96.h5",
        },
        "160": {
            "0.35": "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_160.h5",
            "0.5": "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_160.h5",
            "0.75": "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_160.h5",
            "1": "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5"
        }
    }
}


def download_weights_mobilenetV2(color_channels, input_size, alpha):
    uri = ei_weights_urls.get(str(color_channels), {}).get(str(input_size), {}).get(str(alpha), "")
    if uri == "":
        raise ValueError("Weights not found for values.")

    weights_path = f'./{uri}'
    root_url = 'https://cdn.edgeimpulse.com/'
    p = Path(weights_path)
    if not p.exists():
        print(f"Pretrained weights {weights_path} unavailable; downloading...")
        if not p.parent.exists():
            p.parent.mkdir(parents=True)
        weights_data = requests.get(root_url + weights_path[2:]).content
        with open(weights_path, 'wb') as f:
            f.write(weights_data)
        print(f"Pretrained weights {weights_path} unavailable; downloading OK")
        print("")

    return weights_path


def profile(model_path, device='espressif-esp32'):
    print("Profiling model: " + model_path)
    ei.model.profile(model=model_path, device=device).summary()


if __name__ == "__main__":
    profile("/home/xav/Pictures/sofa/custom_model/trained_classification_quantized.tflite")
