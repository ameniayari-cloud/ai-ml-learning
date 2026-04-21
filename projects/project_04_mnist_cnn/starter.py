"""Starter template for MNIST CNN project."""
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape)
