import os
from os.path import abspath, dirname
import tensorflow as tf
from image_classification.configs.base_config import BaseConfig


class ModelConfig(BaseConfig):
    """
    Model config class that overrides the Baseconfig class
    """
    IMAGE_TYPE = 'png'
    CHANNELS = 1
    resize_shape = (28, 28)
    BATCH_SIZE = 32
    EPOCHS = 10
    VAL_EPOCH = 1
    USE_TFRecords = False
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    OPTIMIZER = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    METRICS = ['accuracy']
    TRAIN_IMAGE_DIR_PATH = os.path.join(dirname(abspath(__file__)), "../../../../data/mnist/images/train/*/*.jpg")
    TEST_IMAGE_DIR_PATH = os.path.join(dirname(abspath(__file__)), "../../../../data/mnist/images/test/*.jpg")
    TFRecords_DIR = os.path.join(dirname(abspath(__file__)), "../../../../data/mnist/tfrecords")
