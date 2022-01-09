import os
import glob
from image_classification.datasets.dataset import Dataset
from image_classification.code.model_ops import ModelOPS
from image_classification.configs.model_config import ModelConfig
from sklearn.model_selection import train_test_split

# comment if you are using CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def execute():
    """Runs training and prediction"""

    ds = Dataset()
    model_op = ModelOPS()

    # Get image paths and labels for training images
    train_images_path = glob.glob(ModelConfig.TRAIN_IMAGE_DIR_PATH)
    test_images_path = glob.glob(ModelConfig.TEST_IMAGE_DIR_PATH)
    train_labels = [int(image_path.split("/")[-2]) for image_path in train_images_path]

    # split training data into train and val set
    train_x, val_x, train_y, val_y = train_test_split(train_images_path, train_labels,
                                                      test_size=0.1, random_state=42)

    # training using TFRecords
    if ModelConfig.USE_TFRecords:
        if not os.path.exists(ModelConfig.TFRecords_DIR):
            os.makedirs(ModelConfig.TFRecords_DIR, exist_ok=True)
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

    # Train model
    model_op.train_on_batches(train_dataset, val_dataset=val_dataset)

    # Predict output using trained model
    probs, preds = model_op.predict_batch(test_dataset)
    print(preds[:12])
    print(probs[:12])

if __name__ == '__main__':
    execute()




