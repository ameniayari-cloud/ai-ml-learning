"""Starter template for CIFAR-10 project."""
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
print(x_train.shape, y_train.shape)
