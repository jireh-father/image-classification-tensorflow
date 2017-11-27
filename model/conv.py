import tensorflow as tf
import json

bn_decay = 0.997
bn_epsilon = 1e-5


def build_model(inputs, num_classes, is_training, model_conf):
    filters = json.loads(model_conf.filters)
    net = inputs
    strides = json.loads(model_conf.strides)
    for i in range(model_conf.num_layers):
        net = tf.layers.conv2d(net, model_conf.filter_size, filters[i], strides=strides[i], padding='valid',
                               kernel_initializer=tf.variance_scaling_initializer(), name="conv" + str(i))

        if model_conf.bn:
            net = tf.layers.batch_normalization(inputs=net, axis=3, momentum=bn_decay, epsilon=bn_epsilon,
                                                center=True, scale=True, training=True, fused=True,
                                                name="batch" + str(i))
        if model_conf.pooling:
            net = tf.layers.max_pooling2d(net, model_conf.pool_size, model_conf.pool_stride, name="maxpool" + str(i))
        net = tf.nn.relu(net, name="relu" + str(i))

        tf.summary.histogram('activations_%d' % i, net)

    output_size = net.get_shape()[1]
    net = tf.layers.conv2d(net, 1, 1, 1, padding='same', kernel_initializer=tf.variance_scaling_initializer,
                           name="last_conv")

    net = tf.reshape(net, [-1, output_size * output_size], name="net_vectorize")
    return tf.layers.dense(net, num_classes, name="dense")


build_model.default_image_size = 224
