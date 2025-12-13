import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("GPU Device Name: ", tf.test.gpu_device_name())
print("Is gpu build", tf.test.is_built_with_gpu_support())
print("Is cuda build", tf.test.is_built_with_cuda())