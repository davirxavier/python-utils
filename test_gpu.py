import os

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("GPU Device Name: ", tf.test.gpu_device_name())
print("Is gpu build", tf.test.is_built_with_gpu_support())
print("Is cuda build", tf.test.is_built_with_cuda())

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Test a small GPU operation
with tf.device('/GPU:0'):
    a = tf.random.uniform([1000, 1000])
    b = tf.matmul(a, a)
print("Tensor computed on GPU:", b.device)