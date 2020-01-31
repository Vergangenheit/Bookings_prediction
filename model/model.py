import tensorflow as tf


class Model:

    def __init__(self, X_train):
        self.input = tf.keras.Input()
        self.lstm = tf.keras.layers.LSTM((None, X_train.shape[1]), recurrent_dropout=0.2)


    def build(self):
        model = self.input
