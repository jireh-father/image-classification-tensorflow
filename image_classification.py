import glob
import os
import sys
from datetime import datetime
import tensorflow as tf
import optimizer
from model import model_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_name', "mnist", "dataset name")
tf.app.flags.DEFINE_string('dataset_dir', "D:\develop\models_new_1122\\research\slim\\mnist_dataset", "dataset_dir")
tf.app.flags.DEFINE_string('log_dir', "log_dir", "save dir")
tf.app.flags.DEFINE_string('model_name', "alexnet_v2", "model name")
tf.app.flags.DEFINE_integer('num_classes', 10, "num_classes")
tf.app.flags.DEFINE_integer('num_channel', 1, "num_channel")
tf.app.flags.DEFINE_integer('batch_size', 16, "batch_size")
tf.app.flags.DEFINE_integer('model_image_size', None, "model_image_size")
tf.app.flags.DEFINE_integer('deconv_image_size', 30, "deconv_image_size")
tf.app.flags.DEFINE_integer('summary_interval', 10, "summary_interval")
tf.app.flags.DEFINE_integer('summary_images', 32, "summary_images")
tf.app.flags.DEFINE_integer('epoch', 100, "epoch")
tf.app.flags.DEFINE_boolean('train', True, "trains")
tf.app.flags.DEFINE_boolean('eval', True, "eval")
tf.app.flags.DEFINE_boolean('bn', False, "bn")
tf.app.flags.DEFINE_boolean('add_image', True, "add_image")
tf.app.flags.DEFINE_integer('add_image_interval', 2, "add_image_interval")
tf.app.flags.DEFINE_boolean('pooling', True, "use max pooling")
tf.app.flags.DEFINE_boolean('pool_size', 3, "pool_size")
tf.app.flags.DEFINE_boolean('pool_stride', 2, "pool_stride")
tf.app.flags.DEFINE_string('strides', "[1,2,1,2,1,2,1]", "strides")
tf.app.flags.DEFINE_integer('filter_size', 64, "filter_size")
tf.app.flags.DEFINE_integer('shuffle_buffer', 50, "shuffle_buffer")
tf.app.flags.DEFINE_integer('num_layers', 7, "deconv layers")
tf.app.flags.DEFINE_integer('num_dataset_parallel', 4, "deconv layers")
tf.app.flags.DEFINE_string('restore_model_path', None, "model path to restore")
tf.app.flags.DEFINE_string('preprocessing_name', None, "preprocessing name")
tf.app.flags.DEFINE_string('filters', "[9, 9, 9,9,9,9,9]", "filters")

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_boolean('cycle_learning_rate', True, "cycle")

NUM_DATASET_MAP = {"mnist": [60000, 10000], "cifar10": [50000, 10000], "flowers": [3320, 350], "block": [4579, 510],
                   "direction": [3036, 332]}
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

model_f = model_factory.get_network_fn(FLAGS.model_name, FLAGS.num_classes, weight_decay=FLAGS.weight_decay,
                                       is_training=is_training)

model_image_size = FLAGS.model_image_size or model_f.default_image_size


def pre_process(example_proto, training):
    features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, features)
    if FLAGS.preprocessing_name:
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocessing_name,
                                                                         is_training=training)
        image = tf.image.decode_image(parsed_features["image/encoded"], FLAGS.num_channel)
        image = tf.clip_by_value(image_preprocessing_fn(image, model_image_size, model_image_size), .0, 1.0)
    else:
        image = tf.clip_by_value(tf.image.per_image_standardization(
            tf.image.resize_images(tf.image.decode_jpeg(parsed_features["image/encoded"], FLAGS.num_channel),
                                   [model_image_size, model_image_size])), .0, 1.0)

    if len(parsed_features["image/class/label"].get_shape()) == 0:
        label = tf.one_hot(parsed_features["image/class/label"], FLAGS.num_classes)
    else:
        label = parsed_features["image/class/label"]

    return image, label


def train_dataset_map(example_proto):
    return pre_process(example_proto, True)


def test_dataset_map(example_proto):
    return pre_process(example_proto, False)


def get_model():
    num_classes = FLAGS.num_classes
    model_name = FLAGS.model_name

    inputs = tf.placeholder(tf.float32, shape=[None, model_image_size, model_image_size, FLAGS.num_channel],
                            name="inputs")

    labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name="labels")
    global_step = tf.Variable(0, trainable=False)
    learning_rate = optimizer.configure_learning_rate(NUM_DATASET_MAP[FLAGS.dataset_name][0], global_step, FLAGS)
    # learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

    if model_name == "deconv":
        logits, gen_x, gen_x_ = model_f(inputs, model_conf=FLAGS)
        class_loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        gen_loss_op = tf.log(
            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gen_x_, logits=gen_x)))
        loss_op = tf.add(class_loss_op, gen_loss_op)

        ops = [class_loss_op, loss_op, gen_loss_op]
        ops_key = ["class_loss_op", "loss_op", "gen_loss_op"]
    else:
        if model_name == "conv":
            logits = model_f(inputs, model_conf=FLAGS)
        else:
            logits, end_points = model_f(inputs)
        if model_name == "resnet":
            logits = tf.reshape(logits, [-1, num_classes])
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        ops = [loss_op]
        ops_key = ["loss_op"]

    tf.summary.scalar('loss', loss_op)
    opt = optimizer.configure_optimizer(learning_rate, FLAGS)
    train_op = opt.minimize(loss_op, global_step=global_step)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy_op)
    merged = tf.summary.merge_all()

    return inputs, labels, train_op, accuracy_op, merged, ops, ops_key


if not os.path.exists(FLAGS.dataset_dir):
    FLAGS.dataset_dir = os.path.join("/home/data", FLAGS.dataset_name)

train_filenames = glob.glob(os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name + "_train*tfrecord"))
test_filenames = glob.glob(os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name + "_validation*tfrecord"))

inputs, labels, train_op, accuracy_op, merged, ops, ops_key = get_model()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if FLAGS.restore_model_path and len(glob.glob(FLAGS.restore_model_path + ".data-00000-of-00001")) > 0:
    saver.restore(sess, FLAGS.restore_model_path)

train_iterator = tf.data.TFRecordDataset(train_filenames).map(train_dataset_map, FLAGS.num_dataset_parallel).shuffle(
    buffer_size=FLAGS.shuffle_buffer).batch(FLAGS.batch_size).make_initializable_iterator()
test_iterator = tf.data.TFRecordDataset(test_filenames).map(test_dataset_map, FLAGS.num_dataset_parallel).batch(
    FLAGS.batch_size).make_initializable_iterator()

num_train = NUM_DATASET_MAP[FLAGS.dataset_name][0]
num_test = NUM_DATASET_MAP[FLAGS.dataset_name][1]
train_step = 0
for epoch in range(FLAGS.epoch):
    if FLAGS.train:
        sess.run(train_iterator.initializer)
        while True:
            try:
                batch_xs, batch_ys = sess.run(train_iterator.get_next())
                results = sess.run([train_op, merged, accuracy_op] + ops,
                                   feed_dict={inputs: batch_xs, labels: batch_ys, is_training: True})
                now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                if train_step % FLAGS.summary_interval == 0:
                    ops_results = " ".join(list(map(lambda x: str(x), list(zip(ops_key, results[3:])))))
                    print(
                        ("[%s TRAIN %d epoch, %d / %d step] accuracy: %f" % (
                            now, epoch, train_step, num_train, results[2])) + ops_results)
                    train_writer.add_summary(results[1], train_step)
                train_step += 1
            except tf.errors.OutOfRangeError:
                break
        saver.save(sess, FLAGS.log_dir + "/model_epoch_%d.ckpt" % epoch)
    if FLAGS.eval:
        total_accuracy = 0
        test_step = 0
        sess.run(test_iterator.initializer)
        while True:
            try:
                test_xs, test_ys = sess.run(test_iterator.get_next())
                results = sess.run(
                    [merged, accuracy_op] + ops, feed_dict={inputs: test_xs, labels: test_ys, is_training: False})
                total_accuracy += results[1]
                now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                ops_results = " ".join(list(map(lambda x: str(x), list(zip(ops_key, results[2:])))))
                print(("[%s TEST %d epoch, %d /%d step] accuracy: %f" % (
                    now, epoch, test_step, num_test, results[1])) + ops_results)
                test_writer.add_summary(results[0], test_step + train_step)
                test_step += 1
            except tf.errors.OutOfRangeError:
                break
        if test_step > 0:
            print("Avg Accuracy : %f" % (float(total_accuracy) / test_step))
    if not FLAGS.train:
        break
