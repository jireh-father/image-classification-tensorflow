import tensorflow as tf
import trainer
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_name', "direction", "dataset name")
tf.app.flags.DEFINE_string('dataset_dir', "F:\data\grading\\direction", "dataset_dir")
tf.app.flags.DEFINE_string('train_name', "train", "train dataset file name")
tf.app.flags.DEFINE_string('test_name', "validation", "test dataset file name")
tf.app.flags.DEFINE_string('log_dir', os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint"),
                           "save dir")
tf.app.flags.DEFINE_integer('vis_epoch', 1, "vis_epoch")
tf.app.flags.DEFINE_integer('num_vis_steps', 10, "num_vis_steps")
tf.app.flags.DEFINE_integer('num_save_interval', 1, "num_save_interval")
tf.app.flags.DEFINE_string('model_name', "alexnet_v2", "model name")
tf.app.flags.DEFINE_integer('batch_size', 16, "batch_size")
tf.app.flags.DEFINE_integer('model_image_size', None, "model_image_size")
tf.app.flags.DEFINE_integer('deconv_image_size', 30, "deconv_image_size")
tf.app.flags.DEFINE_integer('summary_interval', 10, "summary_interval")
tf.app.flags.DEFINE_integer('summary_images', 32, "summary_images")
tf.app.flags.DEFINE_integer('epoch', 3, "epoch")
tf.app.flags.DEFINE_boolean('train', True, "trains")
tf.app.flags.DEFINE_boolean('eval', True, "eval")
tf.app.flags.DEFINE_boolean('bn', False, "bn")
tf.app.flags.DEFINE_boolean('add_image', True, "add_image")
tf.app.flags.DEFINE_integer('add_image_interval', 2, "add_image_interval")
tf.app.flags.DEFINE_boolean('pooling', True, "use max pooling")
tf.app.flags.DEFINE_string('pool_size', "[3,3,3,3,3]", "pool_size")
tf.app.flags.DEFINE_string('pool_stride', "[2,2,2,2,2]", "pool_stride")
tf.app.flags.DEFINE_string('strides', "[1,1,2,2,2]", "strides")
tf.app.flags.DEFINE_string('filter_size', "[64,64,64,64,64]", "filter_size")
tf.app.flags.DEFINE_integer('shuffle_buffer', 50, "shuffle_buffer")
tf.app.flags.DEFINE_integer('num_layers', 5, "deconv layers")
tf.app.flags.DEFINE_integer('num_channel', 1, "num channel")
tf.app.flags.DEFINE_integer('num_dataset_parallel', 4, "deconv layers")
tf.app.flags.DEFINE_string('restore_model_path', None, "model path to restore")
tf.app.flags.DEFINE_string('preprocessing_name', None, "preprocessing name")
tf.app.flags.DEFINE_string('filters', "[9, 9, 9,9,9,9,9]", "filters")
tf.app.flags.DEFINE_float('train_fraction', 0.9, "train_fraction")

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
trainer.train(FLAGS)
