"""Solution for sentiment analysis with Embedding + LSTM."""
import tensorflow as tf

max_words = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_words)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.1)
print(model.evaluate(x_test, y_test, verbose=0))
