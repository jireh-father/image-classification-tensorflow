import tensorflow as tf
from PIL import Image
import numpy as np

img_array = np.array(Image.open("f:/1.jpg"))
data_t = tf.placeholder(tf.uint8)
img = tf.image.encode_jpeg(data_t)
img = tf.image.decode_jpeg(img)
# img = tf.image.resize_image_with_crop_or_pad(img, 500, 500)
img = tf.image.crop_to_bounding_box(img, 0, 0, 200, 200)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

result = sess.run(img, feed_dict={data_t: img_array})
print(result.shape)
img = Image.fromarray(result)
img.save(open("f:/2.jpg", "w+"))
img.close()
