import tensorflow as tf
import os
import glob
from PIL import Image
from datasets.dataset import Dataset
from code.model import MNISTModel
from code.model_ops import ModelOPS
from configs.model_config import ModelConfig
from sklearn.model_selection import train_test_split


def execute():
    ds = Dataset()
    model_op = ModelOPS()

    train_images_path = glob.glob(ModelConfig.TRAIN_IMAGE_DIR_PATH)
    test_images_path = glob.glob(ModelConfig.TEST_IMAGE_DIR_PATH)
    train_labels = [int(image_path.split("/")[-2]) for image_path in train_images_path]

    train_x, val_x, train_y, val_y = train_test_split(train_images_path, train_labels,
                                                      test_size=0.1, random_state=42)

    if ModelConfig.USE_TFRecords:
        train_file = os.path.join(ModelConfig.TFRecords_DIR, 'train.tfrecords')
        val_file = os.path.join(ModelConfig.TFRecords_DIR, 'val.tfrecords')
        test_file = os.path.join(ModelConfig.TFRecords_DIR, 'test.tfrecords')
        train_full = os.path.join(ModelConfig.TFRecords_DIR, 'train_full.tfrecords')
        if not os.path.isfile(train_file):
            ds.write_tf_records(train_x, train_file, labels=train_y)
        if not os.path.isfile(val_file):
            ds.write_tf_records(val_x, val_file, labels=val_y)
        if not os.path.isfile(test_file):
            ds.write_tf_records(test_images_path, test_file)
        if not os.path.isfile(train_full):
            ds.write_tf_records(train_images_path, train_full, labels=train_labels)

        train_dataset, train_dataset_itr = ds.get_tf_record_dataset([train_file])
        val_dataset, val_dataset_itr = ds.get_tf_record_dataset([val_file], op='val')
        test_dataset, test_dataset_itr = ds.get_tf_record_dataset([test_file], has_labels=False, op='test')
        full_train_dataset, full_train_dataset_itr = ds.get_tf_record_dataset([train_full])
    else:
        train_dataset, train_dataset_itr = ds.get_dataset(train_x, labels=train_y)
        val_dataset, val_dataset_itr = ds.get_dataset(val_x, labels=val_y, op='val')
        test_dataset, test_dataset_itr = ds.get_dataset(test_images_path, op='test')
        full_train_dataset, full_train_dataset_itr = ds.get_dataset(train_images_path, labels=train_labels)

    model_op.train_on_batches(train_dataset_itr, val_dataset=val_dataset)

if __name__ == '__main__':
    execute()




