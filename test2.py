import tensorflow as tf
import numpy as np
lr = 0.0001

x = tf.placeholder(tf.float32, [3, 2])
y = tf.placeholder(tf.float32, [3, 1])

h = tf.layers.dense(x, 1)
loss = tf.losses.mean_squared_error(y, h)
# loss = tf.reduce_mean(tf.square(y - h), axis=1)
# print(loss)
op = tf.train.GradientDescentOptimizer(lr)
train = op.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

step = 10

dx = np.array([[3, 2], [4, 3], [5, 4]])
dy = np.array([[6], [8], [10]])

for i in range(step):
    _, loss_result = sess.run([train, loss], feed_dict={x: dx, y: dy})
    print(loss_result)

dx = np.array([[30, 20], [40, 30], [50, 40]])
dy = np.array([[6], [8], [10]])

for i in range(step):
    _, loss_result = sess.run([train, loss], feed_dict={x: dx, y: dy})
    print(loss_result)
