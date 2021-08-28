import tensorflow as tf
from code.model import MNISTModel
from datasets.dataset import Dataset
from configs.model_config import ModelConfig
import numpy as np

class ModelOPS:
    def __init__(self):
        self.dataset = Dataset()
        self.model = MNISTModel()
        self.use_tf_records = ModelConfig.USE_TFRecords

    def get_batch_data(self, batch_data):
        if self.use_tf_records:
            images, labels = self.dataset.restructure_image(batch_data)
        else:
            images, labels = batch_data[0], batch_data[1]
        return images, labels

    def evaluate_on_batches(self, dataset):
        total_correct = 0
        total = 0
        for idx, batch in enumerate(dataset):
            images, labels = self.get_batch_data(batch)
            probs = self.model.predict_on_batch(images)
            preds = tf.argmax(probs, axis=-1)
            correct_labels = tf.argmax(labels, axis=-1)
            is_correct = tf.equal(correct_labels, preds)
            correct = np.sum(is_correct.numpy())
            total_correct += correct
            total += images.shape[0]

        print(f"Total: {total}, Correct: {total_correct}")
        accuracy = total_correct * 100.0 / total
        print(f"Validation Accuracy: {accuracy}")

    def train_on_batches(self, train_dataset, val_dataset=None):
        self.model.compile(
            loss=ModelConfig.LOSS,
            optimizer=ModelConfig.OPTIMIZER,
            metrics=ModelConfig.METRICS
        )
        for idx, batch in enumerate(train_dataset):
            train_images, train_labels = self.get_batch_data(batch)
            history = self.model.train_on_batch(train_images, train_labels)
            if idx % 1000 == 0:
                print(idx, {metric: val for metric, val in zip(self.model.metrics_names, history)})
                if val_dataset:
                    print(f"Calculating validation accuracy after {idx + 1}th iteration:")
                    val_dataset_itr = self.dataset.get_dataset_iterator(val_dataset, op='val')
                    self.evaluate_on_batches(val_dataset)

    def predict_batch(self):
        pass
