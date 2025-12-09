import os
os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\NVIDIA\\CUDNN\\v9.17\\bin\\12.9'

import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Test GPU computation
if tf.config.list_physical_devices('GPU'):
    print("\nTesting GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("GPU matrix multiplication successful!")
else:
    print("\nNo GPU detected")
