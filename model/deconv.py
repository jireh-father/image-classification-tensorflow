import tensorflow as tf
import json

bn_decay = 0.997
bn_epsilon = 1e-5


def build_model(inputs, num_classes, is_training, model_conf):
    filters = json.loads(model_conf.filters)
    strides = json.loads(model_conf.strides)
    net = tf.image.resize_images(inputs, [model_conf.deconv_image_size, model_conf.deconv_image_size])
    for i in range(model_conf.num_layers):
        net = tf.layers.conv2d_transpose(net, model_conf.filter_size, filters[i], strides=strides[i],
                                         padding='valid',
                                         kernel_initializer=tf.variance_scaling_initializer(), name="deconv" + str(i))
        if model_conf.bn:
            net = tf.layers.batch_normalization(inputs=net, axis=3, momentum=bn_decay, epsilon=bn_epsilon,
                                                center=True, scale=True, training=True, fused=True,
                                                name="batch" + str(i))

        net = tf.nn.relu(net, name="relu" + str(i))
        if model_conf.add_image:
            if not model_conf.add_image_interval or (i != 0 and i % model_conf.add_image_interval == 1):
                mid_image = tf.image.resize_images(inputs, [net.get_shape()[1], net.get_shape()[1]])
                if mid_image.get_shape()[3] == 3:
                    mid_image = tf.reduce_mean(mid_image, 3, keep_dims=True)
                net = tf.add(net, mid_image, name="mid_image_" + str(i))
    gen_image_size = net.get_shape()[1]
    gen_x_ = tf.image.resize_images(inputs, [gen_image_size, gen_image_size])
    gen_x_ = tf.reshape(gen_x_, [-1, gen_image_size * gen_image_size], name="gen_y_vectorize")

    gen_x = tf.layers.conv2d(net, model_conf.num_channel, 1, 1, padding='same',
                             kernel_initializer=tf.variance_scaling_initializer, name="gen_x_conv")
    gen_x = tf.nn.sigmoid(gen_x, name="gen_sigmoid")
    tf.summary.image(tensor=gen_x, max_outputs=model_conf.summary_images, name="gen_x")
    gen_x = tf.reshape(gen_x, [-1, gen_image_size * gen_image_size], "gen_x_vectorize")
    net = tf.layers.conv2d(net, 1, 1, 1, padding='same', kernel_initializer=tf.variance_scaling_initializer,
                           name="last_conv")
    net = tf.reshape(net, [-1, gen_image_size * gen_image_size], name="net_vectorize")
    net = tf.layers.dense(net, num_classes, name="dense")

    return net, gen_x, gen_x_


build_model.default_image_size = 224
