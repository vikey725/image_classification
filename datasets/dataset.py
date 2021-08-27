import numpy as np
import tensorflow as tf
import glob

class Dataset:
    def __init__(self, image_paths, image_type, channels, resize_shape=None, filename=None, labels=None):
        self.image_paths = image_paths
        self.image_type = image_type
        self.channels = channels
        self.resize_shape = resize_shape
        self.filename = filename
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

    def decode_img(self, image_path):
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

        return tf.cast(decoded_image, dtype=tf.float32) / 255.0

    def get_one_hot_labels(self):
        one_hot_labels = []
        for label in self.labels:
            one_hot_label = [0] * 10
            one_hot_label[label] = 1
            one_hot_labels.append(one_hot_label)
        return one_hot_labels

    def get_dataset(self):
        filename_tensor = tf.constant(self.image_paths)
        label_tensor = tf.constant(self.get_one_hot_labels())
        dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, label_tensor))
        dataset = dataset.map(lambda x, y: (self.decode_img(x), y))
        return dataset

    def get_dataset_batches(self, batch_size, epoch):
        dataset = self.get_dataset()
        dataset = dataset.shuffle(len(self.labels), reshuffle_each_iteration=True)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch_size)
        dataset = dataset.as_numpy_iterator()
        # for idx, data in enumerate(dataset):
        #     if idx == 10:
        #         break
        #     print(f"Element {idx + 1}: {data[0].shape}, {data[1].shape}")
        return dataset


if __name__ == '__main__':
    image_paths = glob.glob("data/trainingSet/*/*")
    print(len(image_paths))
    labels = [int(image_path.split("/")[-2]) for image_path in image_paths]
    print(len(labels))
    input("...")
    ds = Dataset(image_paths, 'png', 1, labels=labels)

    ds.get_dataset_batches(4)








