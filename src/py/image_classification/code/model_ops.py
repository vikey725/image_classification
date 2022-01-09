import tensorflow as tf
import numpy as np
from image_classification.code.model import MNISTModel
from image_classification.datasets.dataset import Dataset
from image_classification.configs.model_config import ModelConfig


class ModelOPS:
    def __init__(self):
        """
        Supports operations like model training, evaluation, and prediction.
        """
        self.dataset = Dataset()
        self.model = MNISTModel()
        self.use_tf_records = ModelConfig.USE_TFRecords

    def get_batch_data(self, batch_data, op='train'):
        """Separates images and/or labels from Dataset batches"""

        if op == 'train':
            if self.use_tf_records:
                images, labels = self.dataset.restructure_image(batch_data)
            else:
                images, labels = batch_data[0], batch_data[1]
            return images, labels
        else:
            if self.use_tf_records:
                images = self.dataset.restructure_image(batch_data)[0]
            else:
                images = batch_data
            return images

    def evaluate_on_batches(self, dataset):
        """Evaluates dataset batch by batch """

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
        """Trains dataset on batches"""

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
                    if self.use_tf_records:
                        self.evaluate_on_batches(val_dataset)
                    else:
                        self.model.evaluate(val_dataset)
        print(self.model.summary())

    def predict_batch(self, dataset):
        """Predicts dataset on batches anc combines output into a single list"""

        pred_list = []
        score_list = []
        for idx, batch in enumerate(dataset):
            images = self.get_batch_data(batch, op='test')
            # print("Got batch data")
            class_probs = self.model.predict_on_batch(images)
            scores = tf.math.reduce_max(class_probs, axis=-1)
            preds = tf.argmax(class_probs, axis=-1)
            score_list.extend(scores.numpy())
            pred_list.extend(preds.numpy())
        return score_list, pred_list




