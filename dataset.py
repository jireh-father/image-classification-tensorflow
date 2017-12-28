import tensorflow as tf
import os
import random
import math
import sys
import dataset_utils
import json

# Seed for repeatability.
_RANDOM_SEED = 0


def _get_dataset_filename(dataset_name, dataset_dir, split_name, shard_id, num_shards):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (dataset_name, split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)


def _dataset_exists(dataset_name, dataset_dir, num_shards):
    for split_name in ['train', 'validation']:
        for shard_id in range(num_shards):
            output_filename = _get_dataset_filename(
                dataset_name, dataset_dir, split_name, shard_id, num_shards)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def _get_filenames_and_classes(dataset_dir):
    root = dataset_dir
    directories = []
    class_names = []
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, num_channels):
        # Initializes function that decodes Grayscale JPEG data.
        self.num_channels = num_channels
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=num_channels)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == self.num_channels
        return image


def _convert_dataset(dataset_name, split_name, filenames, class_names_to_ids, dataset_dir, num_shards, num_channels=3):
    """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
    assert split_name in ['train', 'validation', 'test']

    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader(num_channels)

        with tf.Session('') as sess:

            for shard_id in range(num_shards):
                output_filename = _get_dataset_filename(dataset_name, dataset_dir, split_name, shard_id, num_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d, %s' % (
                            i + 1, len(filenames), shard_id, filenames[i]))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        if sys.version_info[0] == 3:
                            example = dataset_utils.image_to_tfexample(
                                image_data, 'jpg'.encode(), height, width, class_id)
                        else:
                            example = dataset_utils.image_to_tfexample(
                                image_data, 'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def make_tfrecord(dataset_name, dataset_dir, train_fraction=0.9, num_channels=3, num_shards=4):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_name, dataset_dir, num_shards):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None, None

    random.seed(_RANDOM_SEED)
    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)

    # Divide into train and test:
    print("Now let's start converting the Koreans dataset!")
    random.shuffle(photo_filenames)
    num_train = int(len(photo_filenames) * train_fraction)
    num_validation = int(len(photo_filenames) * (1 - train_fraction))
    num_dataset = len(photo_filenames)
    training_filenames = photo_filenames[:num_train]
    validation_filenames = photo_filenames[num_train:]

    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    # First, convert the training and validation sets.
    _convert_dataset(dataset_name, 'train', training_filenames, class_names_to_ids, dataset_dir, num_shards,
                     num_channels)
    _convert_dataset(dataset_name, 'validation', validation_filenames, class_names_to_ids, dataset_dir, num_shards,
                     num_channels)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    json.dump({"num_train": num_train, "num_validation": num_validation, "num_classes": len(class_names)},
              open(os.path.join(dataset_dir, "metadata"), mode="w+"))

    return num_dataset, len(class_names)

# def tfrecord_sample_nums(dataset_name, dataset_dir):
#     c = 0
#     for fn in tf_records_filenames:
#         for record in tf.python_io.tf_record_iterator(fn):
#             c += 1
