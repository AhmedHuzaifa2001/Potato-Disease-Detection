import numpy as np
import tensorflow as tf

# 1. Create a simple NumPy array
# Let's make a 2x3 array of integers
numpy_array = np.array([[10, 20, 30],
                        [40, 50, 60]])

print("--- NumPy Array ---")
print("Array:\n", numpy_array)
print("Type:", type(numpy_array))
print("Shape:", numpy_array.shape)
print("Data Type (dtype):", numpy_array.dtype)

# 2. Convert the NumPy array to a TensorFlow Tensor
# We can also specify the desired data type for the tensor
tensor_array = tf.convert_to_tensor(numpy_array, dtype=tf.float32)

print("\n--- TensorFlow Tensor ---")
print("Tensor:\n", tensor_array)
print("Type:", type(tensor_array))
print("Shape:", tensor_array.shape)
print("Data Type (dtype):", tensor_array.dtype)

# You can also convert without specifying dtype, TensorFlow will infer it
tensor_array_inferred_dtype = tf.convert_to_tensor(numpy_array)
print("\n--- TensorFlow Tensor (Inferred Dtype) ---")
print("Tensor:\n", tensor_array_inferred_dtype)
print("Data Type (dtype):", tensor_array_inferred_dtype.dtype)