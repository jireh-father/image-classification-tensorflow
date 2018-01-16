import _tkinter
import time
import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import sys
import math
import cv2
from scipy.misc import imread, imresize
import tensorflow as tf
from tensorflow.python.framework import graph_util


class GradCamPlusPlus(object):
    TOP3 = 3
    COLOR_THRESHOLD = 200

    def __init__(self, sess, logits, target_conv_layer, input_tensor):
        assert len(logits.shape) == 2, 'shape of "logits" have to be 2, but {}'.format(len(logits.shape))

        self.sess = sess
        self.target_conv_layer = target_conv_layer
        self.input_tensor = input_tensor

        self.label_vector = tf.placeholder("float", [None, logits.shape[1]])
        self.label_index = tf.placeholder("int64", ())

        cost = logits * self.label_vector
        target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]
        self.first_derivative = tf.exp(cost)[0][self.label_index] * target_conv_layer_grad
        self.second_derivative = tf.exp(cost)[0][self.label_index] * target_conv_layer_grad * target_conv_layer_grad
        self.triple_derivative = tf.exp(cost)[0][
                                     self.label_index] * target_conv_layer_grad * target_conv_layer_grad * target_conv_layer_grad

    def _create_one_hot_encoding(self, vector_size, class_index):
        assert vector_size > class_index, '{}(vector_size) is smaller than {}(class_index)'.format(vector_size,
                                                                                                   class_index)

        output = [0.0] * vector_size
        output[class_index] = 1.0

        return np.array(output)

    def create_cam_img(self, imgs, probs):
        # imgs(RGB, 0~255)
        assert len(imgs.shape) == 4, 'shape of "imgs" have to be (batch size, height, width, channel), but {}'.format(
            imgs.shape)
        assert len(probs.shape) == 2, 'shape of "probs" have to be (batch size, label vector), but {}'.format(
            probs.shape)

        img_height = imgs.shape[1]
        img_width = imgs.shape[2]
        vector_size = probs.shape[1]
        cams = list()
        class_indices = list()

        for i, prob in enumerate(probs):
            sorted_prob = np.argsort(prob)[::-1]

            cams_ = list()
            class_indices_ = list()
            for j in range(GradCamPlusPlus.TOP3):
                ### Create one-hot encoding
                class_index = sorted_prob[j]
                one_hot_encoding = self._create_one_hot_encoding(vector_size, class_index)
                class_indices_.append(class_index)

                ### Input image Normalization
                img = imgs[i]
                if imgs[i].min() < 0:
                    img = imgs[i] - imgs[i].min()
                    img /= img.max()
                assert (0 <= img).all() and (img <= 1.0).all()

                ### Create CAM image
                conv_output, conv_first_grad, conv_second_grad, conv_third_grad = self.sess.run(
                    [self.target_conv_layer, self.first_derivative, self.second_derivative, self.triple_derivative],
                    feed_dict={self.input_tensor: [img], self.label_index: class_index,
                               self.label_vector: [one_hot_encoding]})
                global_sum = np.sum(conv_output[0].reshape((-1, conv_first_grad[0].shape[2])), axis=0)
                alpha_num = conv_second_grad[0]
                alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape(
                    (1, 1, conv_first_grad[0].shape[2]))
                alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
                alphas = alpha_num / alpha_denom
                weights = np.maximum(conv_first_grad[0], 0.0)
                alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
                alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))
                deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])),
                                                    axis=0)
                grad_cam_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)
                cam = np.maximum(grad_cam_map, 0)
                cam = cam / np.max(cam)  # scale 0 to 1.0
                cam = resize(cam, (img_height, img_width))

                ### CAM image Denormalization
                cam = np.uint8(cam * 255)  # denomalization

                cams_ = [cam] if len(cams_) == 0 else np.append(cams_, [cam], axis=0)
            cams = [cams_] if len(cams) == 0 else np.append(cams, [cams_], axis=0)
            class_indices.append(class_indices_)

        return cams, class_indices  # cams: (batch size, top 1~3, height, width), class_indices: (batch size, top 1~3)

    def convert_cam_2_heatmap(self, cam):
        return cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    def overlay_heatmap(self, img, heatmap):
        return cv2.addWeighted(heatmap, 0.4, img, 0.5, 0)

    def _get_upper_boundary(self, img, height, width):
        for h in range(height):
            for w in range(width):
                if int(img[h, w]) > GradCamPlusPlus.COLOR_THRESHOLD:
                    return 0 if h == 0 else h - 1

    def _get_lower_boundary(self, img, height, width):
        for h in reversed(range(height)):
            for w in range(width):
                if int(img[h, w]) > GradCamPlusPlus.COLOR_THRESHOLD:
                    return height if h == height else h + 1

    def draw_rectangle(self, img, cam, box_color):
        # box_color : rgb

        ### Get height, width
        height, width = cam.shape

        ### Get top, bottom, left, right
        top = self._get_upper_boundary(cam, height, width)
        bottom = self._get_lower_boundary(cam, height, width)
        transpose_img = np.transpose(cam)
        left = self._get_upper_boundary(transpose_img, width, height)
        right = self._get_lower_boundary(transpose_img, width, height)

        return cv2.rectangle(img, (left, top), (right, bottom), tuple(box_color[::-1]), 3)  # color : bgr!!!!
