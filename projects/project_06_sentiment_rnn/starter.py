"""Starter template for sentiment analysis with RNN."""
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.imdb.load_data(num_words=10000)
print(len(x_train), len(y_train))
