import tensorflow as tf


class BaseConfig:
    IMAGE_TYPE = 'jpeg'
    CHANNELS = 3
    resize_shape = (28, 28)
    BATCH_SIZE = 8
    EPOCHS = 20
    USE_TFRecords = False
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    OPTIMIZER = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    METRICS = ['accuracy']
    TFRecords_DIR = "data"
