import tensorflow as tf

class Dataset:
    def __init__(self, image_paths, filename, image_type, channels, resize_shape, labels=None):
        self.image_paths = image_paths
        self.filename = filename
        self.image_type = image_type
        self.channels = channels
        self.resize_shape = resize_shape
        self.labels = labels

    @staticmethod
    def convert_to_int_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def convert_to_float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def convert_to_byte_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def decode_image(self, image_path):
        value = tf.io.read_file(image_path)
        if self.image_type == 'gif':
            return ValueError("ERROR, .gif not supported.")
        if self.image_type == 'png':
            decoded_image = tf.io.decode_png(value, channels=self.channels)
        elif self.image_type == 'jpeg':
            decoded_image = tf.io.decode_jpeg(value, channels=self.channels)
        else:
            decoded_image = tf.io.decode_image(value, channels=self.channels)

        if self.resize_shape:
            decoded_image = tf.image.resize(decoded_image, self.resize_shape)

        return decoded_image







