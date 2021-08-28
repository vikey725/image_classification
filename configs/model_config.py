import tensorflow as tf
from configs.base_config import BaseConfig


class ModelConfig(BaseConfig):
    IMAGE_TYPE = 'png'
    CHANNELS = 1
    resize_shape = (28, 28)
    BATCH_SIZE = 32
    EPOCHS = 10
    VAL_EPOCH = 1
    USE_TFRecords = True
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    OPTIMIZER = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    METRICS = ['accuracy']
    TRAIN_IMAGE_DIR_PATH = "data/mnist/images/train/*/*.jpg"
    TEST_IMAGE_DIR_PATH = "data/mnist/images/test/*.jpg"
    TFRecords_DIR = "data/mnist/tfrecords"
