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

    def call(self, input_tensor):
        layer = self.conv1(input_tensor)
        layer = self.pool1(layer)
        layer = self.flatten(layer)
        layer = self.dense1(layer)
        layer = self.dropout1(layer)
        return self.dense2(layer)


if __name__ == '__main__':
    model = MNISTModel()
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
        metrics=['accuracy']
    )

    image_paths = glob.glob("data/trainingSet/*/*")
    print(len(image_paths))
    labels = [int(image_path.split("/")[-2]) for image_path in image_paths]
    print(len(labels))
    ds = Dataset(image_paths, 'png', 1, labels=labels)

    batch_dataset = ds.get_dataset_batches(32, 10)
    print("Entering loop")
    for idx, batch in enumerate(batch_dataset):
        history = model.train_on_batch(batch[0], batch[1])
        if idx % 500 == 0:
            print(idx, {metric: val for metric, val in zip(model.metrics_names, history)})


