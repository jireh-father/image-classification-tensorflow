import tensorflow as tf
from PIL import Image
import numpy as np

img_array = np.array(Image.open("f:/1.jpg"))
data_t = tf.placeholder(tf.uint8)
img = tf.image.encode_jpeg(data_t)
img = tf.image.decode_jpeg(img)
# img = tf.image.resize_image_with_crop_or_pad(img, 500, 500)
# img = tf.image.crop_to_bounding_box(img, 0, 0, 200, 200)
print(img.get_shape(), type(img.get_shape()))
image_width = int(img.get_shape()[1])
image_height = int(img.get_shape()[2])
if image_width > image_height:
    crop_length = image_height
    crop_x_offset = image_width / 2 - crop_length / 2
    crop_y_offset = 0
else:
    crop_length = image_width
    crop_x_offset = 0
    crop_y_offset = image_height / 2 - crop_length / 2

image = tf.image.crop_to_bounding_box(img, crop_x_offset, crop_y_offset, crop_length, crop_length)
distorted_image = tf.image.resize_images(image, [299, 299])
sess = tf.Session()
sess.run(tf.global_variables_initializer())

result = sess.run(img, feed_dict={data_t: img_array})
print(result.shape)
img = Image.fromarray(result)
img.save(open("f:/2.jpg", "w+"))
img.close()
