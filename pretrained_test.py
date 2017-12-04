import tensorflow as tf
from model import model_factory
import json
from PIL import Image
import numpy as np
import glob


def image_to_vector(path):
    img = Image.open(path)
    return np.array(img)


model_name = "inception_resnet_v2"
model_image_size = 299
num_channel = 3
num_classes = 1001
inputs = tf.placeholder(tf.float32, shape=[None, model_image_size, model_image_size, num_channel],
                        name="inputs")

model_f = model_factory.get_network_fn(model_name, num_classes, weight_decay=0.00004, is_training=False)

logits, end_points = model_f(inputs)

merged = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
writer = tf.summary.FileWriter("pretrained_test", sess.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "D:\data\model-trained\inception_resnet_v2_2016_08_30.ckpt")

images = glob.glob("F:\data\edibot\image3/*")
images.sort()
data = []
for image in images:
    data.append(image_to_vector(image))
data = np.array(data)
data = data / data.max()
result = sess.run(logits, feed_dict={inputs: data})
print(result.shape)
json.dump(result.tolist(), open("pretrain_feature.json", mode="w+"))
