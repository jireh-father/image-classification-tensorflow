import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lr = 0.001
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

x = tf.placeholder(tf.float32, [1, 2])
y = tf.placeholder(tf.float32, [1, 1])

z = tf.layers.dense(x, 2, name="l1")
bm1 = tf.layers.batch_normalization(
    inputs=z, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=True, fused=True)
h = tf.layers.dense(bm1, 1, name="l2")
bm2 = tf.layers.batch_normalization(
    inputs=h, axis=1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=True, fused=True)

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

step = 10

# dx = np.array([[3, 2], [4, 3], [5, 4]])
# dy = np.array([[6], [8], [10]])
#
# for i in range(step):
#     _, loss_result = sess.run([train, loss], feed_dict={x: dx, y: dy})
#     print(loss_result)

dx = np.array([[30, 20]])
dy = np.array([[6]])

# dx = np.array([[30.1, 30.]])
# dy = np.array([[6]])

# dx = np.array([[.30, .20]])
# dy = np.array([[6]])
xx = []
yy = []
zz = []
for i in range(step):
    _, loss_result, weights, gradients1, bm1_r, bm2_r, z_r, h_r = sess.run([train, loss, w, grad1, bm1, bm2, z, h],
                                                                           feed_dict={x: dx, y: dy})
    print("=======================================================================")
    print("# step %d" % i)
    print("loss", loss_result)
    print("weights", weights)
    print("grad", gradients1)
    print("gard variance", gradients1.var())

    zz.append(loss_result)
    xx.append(weights[0][0][0])
    yy.append(weights[0][1][0])
    print("w1 update", weights[0][0][0], -(gradients1[0][0] * lr))
    print("w2 update", weights[0][1][0], -(gradients1[1][0] * lr))
    print("z and h variance", z_r.var(), h_r.var())
    print("bm1 and bm2 variance", bm1_r.var(), bm2_r.var())
    # Axes3D.plot_trisurf(gradients1[0][0], gradients1[1][0], loss_result)
    # print(gradients2)
    # print(gradients2.var())

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# c = ['red', 'blue', 'green', 'yellow', 'gray', 'black', 'cyan', 'magenta', 'orange', 'purple']
# # xx = np.array(xx)
# # yy = np.array(yy)
# # zz = np.array(zz)
# # xx = (xx - xx.min()) / (xx.max() - xx.min())
# # yy = (yy - yy.min()) / (yy.max() - yy.min())
# # zz = (zz - zz.min()) / (zz.max() - zz.min())
# print(xx)
# print(yy)
# print(zz)
# for i in range(len(xx) - 1):
#     ax.plot([xx[i], xx[i + 1]], [yy[i], yy[i + 1]], [zz[i], zz[i + 1]], c=c[i])  # , )
# # ax.scatter(xx, yy, zz)
# # ax.plot([1,2],[1,2],[1,2])
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()
