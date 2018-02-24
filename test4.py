import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lr = 1

x = tf.placeholder(tf.float32, [1, 2])
y = tf.placeholder(tf.float32, [1, 1])

# z = tf.layers.dense(x, 2, name="l1", use_bias=False, kernel_initializer=tf.initializers.ones)
h = tf.layers.dense(x, 1, name="l2", use_bias=False, kernel_initializer=tf.initializers.ones)

w = tf.trainable_variables()

loss = tf.losses.mean_squared_error(y, h)

grad1 = tf.gradients(loss, w[0])[0]
# grad2 = tf.gradients(loss, w[1])[0]
# loss = tf.reduce_mean(tf.square(y - h), axis=1)
# print(loss)
op = tf.train.GradientDescentOptimizer(lr)
train = op.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

step = 8

# dx = np.array([[3, 2], [4, 3], [5, 4]])
# dy = np.array([[6], [8], [10]])
#
# for i in range(step):
#     _, loss_result = sess.run([train, loss], feed_dict={x: dx, y: dy})
#     print(loss_result)

# dx = np.array([[30, 20]])
# dy = np.array([[6]])

dx = np.array([[.30, .20]])
dy = np.array([[6]])
xx = []
yy = []
zz = []
import math
for i in range(step):
    _, loss_result, weights, gradients1= sess.run([train, loss, w, grad1], feed_dict={x: dx, y: dy})
    print(loss_result)
    print(weights)
    print(gradients1)
    print(gradients1.var())
    zz.append(loss_result)
    xx.append(weights[0][0][0])
    yy.append(weights[0][1][0])
    # Axes3D.plot_trisurf(gradients1[0][0], gradients1[1][0], loss_result)
    # print(gradients2)
    # print(gradients2.var())

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xx, yy, zz)#, c=['red', 'blue', 'green', 'yellow', 'gray', 'black', 'cyan', 'magenta'])
# ax.plot_trisurf(xx, yy, zz, linewidth=20, antialiased=True)

plt.show()