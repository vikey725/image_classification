import numpy as np
import tensorflow as tf
from PIL import Image
from image_classification.configs.model_config import ModelConfig


class Dataset:
    def __init__(self):
        """
        Creates image dataset iterator using tf.data.Dataset.from_tensor_slices and tfrcords
        """
        self.image_type = ModelConfig.IMAGE_TYPE
        self.channels = ModelConfig.CHANNELS
        self.resize_shape = ModelConfig.resize_shape
        self.feature_map = None
        self.has_labels = False

    @staticmethod
    def convert_to_int_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def convert_to_float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def convert_to_byte_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def decode_img(self, image_path, normalize=True):
        """
        Function to read the image, resize and normalize it
        Args:
            image_path (str): Path of the image
            normalize (bool): normalizes the images if true

        """
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

        decoded_image = tf.cast(decoded_image, dtype=tf.float32)
        if normalize:
            return self.normalize_image(decoded_image)
        return decoded_image

    def read_image(self, image_path, image_mode=None, normalize=True):
        """
        Read image using python PIL library and
        Args:
            image_path (str): path of the images
            image_mode (str): the requested mode. See: Modes.
            normalize (bool): normalizes the image if True

        """
        img = Image.open(image_path)
        if image_mode:
            img = img.convert(image_mode)
        img = img.resize(self.resize_shape, Image.LANCZOS)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = img.reshape((img.shape[0], img.shape[1], 1))
        if normalize:
            return self.normalize_image(img)
        return img

    @staticmethod
    def normalize_image(image):
        """Normalizes images"""
        return image / 255.0

    @staticmethod
    def get_one_hot_labels(labels):
        """
        Args:
            labels (list): list of labels

        Returns: one hot encoding for given labels; (list<list>)

        """
        one_hot_labels = []
        for label in labels:
            one_hot_label = [0] * 10
            one_hot_label[label] = 1
            one_hot_labels.append(one_hot_label)
        return one_hot_labels

    def get_dataset(self, image_paths, labels=None, op='train'):
        """
        for given image paths and labels, returns corresponding dataset object and its numpy iterator
        Args:
            image_paths (list<str>): list of paths of images
            labels (list<list>): one hot encoding of labels
            op (str): operation type; train, val etc

        """
        filename_tensor = tf.constant(image_paths)
        if labels:
            label_tensor = tf.constant(self.get_one_hot_labels(labels))
            dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, label_tensor))
            dataset = dataset.map(lambda x, y: (self.decode_img(x), y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)
            dataset = dataset.map(lambda x: self.decode_img(x))
        dataset, dataset_itr = self.get_dataset_iterator(dataset, op)
        return dataset, dataset_itr

    @staticmethod
    def get_dataset_iterator(dataset, op):
        """
        For a given dataset, returns dataset and its corresponding numpy iterator
        Args:
            dataset: tf.data.Dataset object
            op: operation type; train, val etc.

        """
        epochs = ModelConfig.EPOCHS
        if op != 'train':
            epochs = ModelConfig.VAL_EPOCH
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(ModelConfig.BATCH_SIZE)
        dataset_itr = dataset.as_numpy_iterator()
        return dataset, dataset_itr

    def get_serialize_example(self, img, one_hot_label=None):
        """
        converts data into a serialized format to store as tfRecords
        Args:
            img (numpy): image as numpy array
            one_hot_label (list<list>): one hot encoded labels
        """
        feature = {
            'images': Dataset.convert_to_byte_feature(img.tobytes()),
            'width': Dataset.convert_to_int_feature(self.resize_shape[0]),
            'height': Dataset.convert_to_int_feature(self.resize_shape[1]),
            'channels': Dataset.convert_to_int_feature(self.channels)
        }
        if one_hot_label is not None:
            feature['one_hot_labels'] = Dataset.convert_to_byte_feature(one_hot_label.tobytes())
        image_example = tf.train.Example(features=tf.train.Features(feature=feature))
        ser_image_example = image_example.SerializeToString()
        return ser_image_example

    def write_tf_records(self, image_paths, filename, labels=None):
        """
        Writes data as tf records
        Args:
            image_paths (list<str>): list of image paths
            filename (str): name of tfRecord file
            labels (list<list>): one hot encoded labels

        """
        writer = tf.io.TFRecordWriter(filename)
        if labels:
            one_hot_labels = self.get_one_hot_labels(labels)
            for image_path, one_hot_label in zip(image_paths, one_hot_labels):
                img = self.read_image(image_path)
                ser_example = self.get_serialize_example(img, np.array(one_hot_label))
                writer.write(ser_example)
        else:
            for image_path in image_paths:
                img = self.read_image(image_path)
                ser_example = self.get_serialize_example(img)
                writer.write(ser_example)
        writer.close()

    def parse_example(self, example):
        """
        Parses tf.train.Example back to image and labels

        """
        data = tf.io.parse_example(example, self.feature_map)
        image = tf.io.decode_raw(data['images'], tf.float64)
        image = tf.cast(image, tf.float32)
        one_hot_label = None
        if self.has_labels:
            one_hot_label = tf.io.decode_raw(data['one_hot_labels'], tf.int64)
        width = tf.cast(data['width'], tf.int64)
        height = tf.cast(data['height'], tf.int64)
        data_dict = {
            'images': image,
            'width': width,
            'height': height
        }
        if self.has_labels:
            data_dict['labels'] = one_hot_label
        return data_dict

    def get_tf_record_dataset(self, filenames, has_labels=True, op='train'):
        """
        Reads tfRecords file as tf.data.TFRecordDataset
        Args:
            filenames (str): tfRecord file name
            has_labels (bool): labels present or not
            op (str): operation type; train, test etc

        """
        self.has_labels = has_labels
        dataset = tf.data.TFRecordDataset(filenames)
        self.feature_map = {
            'images': tf.io.FixedLenFeature((), tf.string),
            'width': tf.io.FixedLenFeature((), tf.int64),
            'height': tf.io.FixedLenFeature((), tf.int64),
            'channels': tf.io.FixedLenFeature((), tf.int64)
        }
        if self.has_labels:
            self.feature_map['one_hot_labels'] = tf.io.FixedLenFeature((), tf.string)
        dataset = dataset.map(self.parse_example)
        dataset, dataset_itr = self.get_dataset_iterator(dataset, op)
        return dataset, dataset_itr

    @staticmethod
    def restructure_image(batch_dict, op='train'):
        """
        Reconstructs the image back to desired format
        Args:
            batch_dict : tf.data.TFRecordDataset batch
            op (str): operation type; train, test etc

        Returns:

        """
        width = batch_dict['width'][0]
        height = batch_dict['height'][0]
        images = batch_dict['images']
        images = tf.reshape(images, (images.shape[0], width, height, -1))
        if op == 'train':
            one_hot_labels = batch_dict.get('labels', None)
            return images, one_hot_labels
        return images
