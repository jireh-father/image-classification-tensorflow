import tensorflow as tf
import os
import random

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


def make_tfrecord(dataset_name, dataset_dir, train_fraction=0.9, num_shards=4):
    pass
    # if not tf.gfile.Exists(dataset_dir):
    #     tf.gfile.MakeDirs(dataset_dir)
    #
    # if _dataset_exists(dataset_dir, num_shards):
    #     print('Dataset files already exist. Exiting without re-creating them.')
    #     return
    #
    # random.seed(_RANDOM_SEED)
    # photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    #
    # # Divide into train and test:
    # print("Now let's start converting the Koreans dataset!")
    # random.shuffle(photo_filenames)
    # training_filenames = photo_filenames[FLAGS.num_validation:]
    # validation_filenames = photo_filenames[:FLAGS.num_validation]
    #
    # class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    # # First, convert the training and validation sets.
    # _convert_dataset('train', training_filenames, class_names_to_ids,
    #                  dataset_dir, output_suffix)
    # _convert_dataset('validation', validation_filenames, class_names_to_ids,
    #                  dataset_dir, output_suffix)
    #
    # # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    # if output_suffix:
    #     dataset_utils.write_label_file(labels_to_class_names, dataset_dir, 'labels_' + output_suffix + '.txt')
    # else:
    #     dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
