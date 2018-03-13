import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

dataset_dir = "mnist_test"
batch_size = 16
lr = 0.01
input_dimension = 784

x = tf.placeholder(tf.float32, [None, input_dimension])
y = tf.placeholder(tf.int32, [None, 10])

z1 = tf.layers.dense(x, input_dimension / 2)
a1 = tf.nn.relu(z1)

z2 = tf.layers.dense(x, 10)
# a2 = tf.nn.relu(z2)
#
# z3 = tf.layers.dense(x, input_dimension)
# a3 = tf.nn.relu(z3)
#
# z4 = tf.layers.dense(x, input_dimension)
# a4 = tf.nn.relu(z4)
#
# z5 = tf.layers.dense(x, input_dimension)
# a5 = tf.nn.relu(z5)

w = tf.trainable_variables()

loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(y, z2))
grad = tf.gradients(loss, w)
op = tf.train.GradientDescentOptimizer(lr)
train = op.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

step = 10000

data_sets = input_data.read_data_sets(dataset_dir, one_hot=True)

for i in range(step):
    print("%d : =====================================" % i)
    train_x, train_y = data_sets.train.next_batch(batch_size)
    train_x *= 100
    # train_x = train_x - 0.5 * 2
    print("x mean", train_x.mean())
    print("x var", train_x.var())

    _, z_, a_, z2_, w_, loss_ = sess.run([train, z1, a1, z2, w, loss], feed_dict={x: train_x, y: train_y})
    print("loss", loss_)
    print("z mean", z_.mean())
    print("z var", z_.var())

    print("a mean", a_.mean())
    print("a var", a_.var())

    print("z2 mean", z2_.mean())
    print("z2 var", z2_.var())
    # print("w1 mean", w_[0].mean())
    # print("w1 var", w_[0].var())
    # print("w2 mean", w_[1].mean())
    # print("w2 var", w_[1].var())
    # print("w3 mean", w_[2].mean())
    # print("w3 var", w_[2].var())
    # print("w4 mean", w_[3].mean())
    # print("w4 var", w_[3].var())

    # print(grad_.shape)
