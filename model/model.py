import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from config import config
import numpy as np
import pandas as pd
import os
from sklearn.utils import class_weight


def cls_weights(sessions_merged: pd.DataFrame) -> np.array:
    """Args: sessions dataset
    Returns: array of class weights to be used in the training cycle"""

    class_weights = class_weight.compute_class_weight('balanced', sessions_merged['has_booking'].unique(),
                                                      sessions_merged['has_booking'].values)

    return class_weights


class Model:

    def __init__(self, no_unique_actions):
        self.model = tf.keras.Sequential()
        self.no_unique_actions = no_unique_actions
        self.embedding_size = int(min(np.ceil(self.no_unique_actions / 2), 50))
        self.optimizer = Adam(lr=0.001)

    def build(self):
        self.model.add(Embedding(self.no_unique_actions + 1, self.embedding_size))
        self.model.add(LSTM(self.embedding_size, return_sequences=True, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))

    def train(self, X_train, y_train, X_val, y_val, class_weights):
        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=self.optimizer)
        self.model.fit(X_train, y_train, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=2,
                       validation_data=(X_val, y_val),
                       class_weight=dict(enumerate(class_weights)))

    def save_model(self):
        self.model.save(config.MODEL_PATH)
