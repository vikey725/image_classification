import tensorflow as tf
import glob
import numpy as np
from datasets.dataset import Dataset
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout
)
from tensorflow.keras import Model

tf.config.experimental_run_functions_eagerly(True)

class MNISTModel(Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.pool1 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(10, activation='softmax')

    def call(self, input_tensor, is_training=True, mask=None):
        layer = self.conv1(input_tensor)
        layer = self.pool1(layer)
        layer = self.flatten(layer)
        layer = self.dense1(layer)
        if is_training:
            layer = self.dropout1(layer)
        return self.dense2(layer)

