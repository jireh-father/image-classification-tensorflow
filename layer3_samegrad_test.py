import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lr = 0.001
weight_decay = 0.1
x = tf.placeholder(tf.float64, [200, 2])
y = tf.placeholder(tf.float64, [200, 1])

z1 = tf.layers.dense(x, 2, name="l1", use_bias=True, kernel_initializer=tf.initializers.zeros)
a1 = tf.nn.relu(z1)
z2 = tf.layers.dense(a1, 2, name="l2", use_bias=True, kernel_initializer=tf.initializers.zeros)
a2 = tf.nn.relu(z2)
h = tf.layers.dense(a2, 1, name="l3", use_bias=True, kernel_initializer=tf.initializers.zeros)

weights = tf.trainable_variables()
loss = tf.losses.mean_squared_error(y, h)

grad = tf.gradients(loss, weights)

# grad2 = tf.gradients(loss, w[1])[0]
# loss = tf.reduce_mean(tf.square(y - h), axis=1)
# print(loss)
op = tf.train.GradientDescentOptimizer(lr)
train = op.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

step = 10

# dx = np.random.randn(10, 2)
# dy = np.random.randn(10, 10)
xxxx = []
yyyy = []
import random

for i in range(100):
    for j in range(2):
        xxxx.append(np.array([i + j, i + j + 1]) / 100.)
        yyyy.append([float(i + j * 2 / 100) + (random.random() - 0.5)])
dx = np.array([[3, 2], [4, 3], [5, 4]])
dx = np.array(xxxx)
# dx = (dx - 0.5) * 2
# print(dx)
dy = np.array([[6], [8], [10]])
dy = np.array(yyyy)
#
# for i in range(step):
#     _, loss_result = sess.run([train, loss], feed_dict={x: dx, y: dy})
#     print(loss_result)

# dx = np.array([[30, 20]])
# dy = np.array([[6]])

# dx = np.array([[30.1, 30.]])
# dy = np.array([[6]])

# dx = np.array([[.30, .20]])
# dy = np.array([[6]])
xx = []
yy = []
zz = []

for i in range(step):
    _, loss_result, weights_result, gradients, t1, t2, t3, t4, t5 = sess.run(
        [train, loss, weights, grad, z1, z2, h, a1, a2], feed_dict={x: dx, y: dy})
    print("=======================================================================")
    # print("# step %d" % i)
    print("loss", loss_result)
    # print("weights", weights_result)
    # print(gradients)
    print(gradients[0])
    print("=")
    print(gradients[1])
    print("=")
    print(gradients[2])
    print("=")
    # print(t1)
    # print("=")
    # print(t2)
    # print("=")
    # print(t3)
    # print("=")
    # print(t4)
    # print("=")
    # print(t5)
    # print(gradients.shape)
    # print("grad1", gradients[0])
    # print("gard variance1", gradients[0].var())
    # print("grad2", gradients[1])
    # print("gard variance2", gradients[1].var())
    # print("grad3", gradients[2])
    # print("gard variance3", gradients[2].var())
    # zz.append(loss_result)
    # xx.append(weights_result[0][0][0])
    # yy.append(weights_result[0][1][0])
    # print("w11 update", weights_result[0][0][0], -(gradients[0][0][0] * lr))
    # print("w12 update", weights_result[0][1][0], -(gradients[0][1][0] * lr))

    # Axes3D.plot_trisurf(gradients1[0][0], gradients1[1][0], loss_result)
    # print(gradients2)
    # print(gradients2.var())

fig = plt.figure()
ax = fig.gca(projection='3d')
c = ['red', 'blue', 'green', 'yellow', 'gray', 'black', 'cyan', 'magenta', 'orange', 'purple']
# xx = np.array(xx)
# yy = np.array(yy)
# zz = np.array(zz)
# xx = (xx - xx.min()) / (xx.max() - xx.min())
# yy = (yy - yy.min()) / (yy.max() - yy.min())
# zz = (zz - zz.min()) / (zz.max() - zz.min())
print(xx)
print(yy)
print(zz)
# for i in range(len(xx) - 1):
#     ax.plot([xx[i], xx[i + 1]], [yy[i], yy[i + 1]], [zz[i], zz[i + 1]], c=c[i])  # , )
# # ax.scatter(xx, yy, zz)
# # ax.plot([1,2],[1,2],[1,2])
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()
