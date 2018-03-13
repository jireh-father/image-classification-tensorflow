import numpy as np
import sys
import cv2
import tensorflow as tf


class GradCamPlusPlus(object):
    TOP3 = 1
    COLOR_THRESHOLD = 200

    def __init__(self, logit, last_conv_layer, input_tensor, is_training):
        self._build_net(logit, last_conv_layer, input_tensor, is_training)

    def _build_net2(self, logit, last_conv_layer, input_tensor):
        assert len(logit.shape) == 2, 'len(logit.shape) == 2, but len(logit.shape): {}'.format(len(logit.shape))

        self.last_conv_layer = last_conv_layer
        self.input_tensor = input_tensor
        self.label_vector = tf.placeholder("float", [None, logit.shape[1]])
        self.label_index = tf.placeholder("int64")
        self.is_training = is_training

        cost = logit * self.label_vector
        last_conv_layer_grad = tf.gradients(cost, last_conv_layer)[0]
        self.cost_gard = last_conv_layer_grad[0]

    def _build_net(self, logit, last_conv_layer, input_tensor, is_training):
        assert len(logit.shape) == 2, 'len(logit.shape) == 2, but len(logit.shape): {}'.format(len(logit.shape))

        self.last_conv_layer = last_conv_layer
        self.input_tensor = input_tensor
        self.label_vector = tf.placeholder("float", [None, logit.shape[1]])
        self.label_index = tf.placeholder("int64")
        self.is_training = is_training
        cost = logit * self.label_vector

        last_conv_layer_grad = tf.gradients(cost, last_conv_layer)[0]
        self.first_derivative = tf.exp(cost)[0][self.label_index] * last_conv_layer_grad
        self.second_derivative = tf.exp(cost)[0][self.label_index] * last_conv_layer_grad * last_conv_layer_grad
        self.triple_derivative = tf.exp(cost)[0][
                                     self.label_index] * last_conv_layer_grad * last_conv_layer_grad * last_conv_layer_grad

    def _create_one_hot_encoding(self, vector_size, class_index):
        assert vector_size > class_index, '{}(vector_size) is smaller than {}(class_index)'.format(vector_size,
                                                                                                   class_index)

        output = [0.0] * vector_size
        output[class_index] = 1.0

        return np.array(output)

    def _create_cam_img2(self, sess, img, class_index, one_hot_encoding):
        conv_output, conv_grad = sess.run(
            [self.last_conv_layer, self.cost_gard],
            feed_dict={self.input_tensor: [img], self.label_index: class_index,
                       self.label_vector: [one_hot_encoding], self.is_training: False})

        feature_map = conv_output[0] * conv_grad
        sum_feature_map = np.sum(feature_map, axis=2)
        cam = (sum_feature_map - sum_feature_map.min()) / (sum_feature_map.max() - sum_feature_map.min())

        return cv2.resize(cam, (img.shape[0], img.shape[1]))

    def _create_cam_img3(self, sess, img, class_index, one_hot_encoding):
        conv_output, conv_grad = sess.run(
            [self.last_conv_layer, self.cost_gard],
            feed_dict={self.input_tensor: [img], self.label_index: class_index,
                       self.label_vector: [one_hot_encoding], self.is_training: False})
        global_sum = np.sum(conv_grad.reshape((-1, conv_grad.shape[2])), axis=0)
        feature_map = conv_output[0] * global_sum.reshape((1, 1, conv_grad.shape[2]))
        sum_feature_map = np.sum(feature_map, axis=2)
        cam = (sum_feature_map - sum_feature_map.min()) / (sum_feature_map.max() - sum_feature_map.min())

        return cv2.resize(cam, (img.shape[0], img.shape[1]))

    def _create_cam_img(self, sess, img, class_index, one_hot_encoding):
        conv_output, conv_first_grad, conv_second_grad, conv_third_grad = sess.run(
            [self.last_conv_layer, self.first_derivative, self.second_derivative, self.triple_derivative],
            feed_dict={self.input_tensor: [img], self.label_index: class_index, self.label_vector: [one_hot_encoding],
                       self.is_training: False})
        global_sum = np.sum(conv_output[0].reshape((-1, conv_first_grad[0].shape[2])), axis=0)
        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape(
            (1, 1, conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num / alpha_denom
        weights = np.maximum(conv_first_grad[0], 0.0)
        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)
        alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))
        deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)
        grad_cam_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)
        cam = np.maximum(grad_cam_map, 0)
        cam = cam / np.max(cam)
        return cv2.resize(cam, (img.shape[0], img.shape[1]))

    def _normalize_images(self, imgs):
        if imgs.max() > 1:
            imgs = imgs / 255.0

        return imgs

    def create_cam_imgs(self, sess, imgs, probs):
        assert len(probs.shape) == 2, 'len(probs.shape) == 2 (batch size, label vector), but {}'.format(
            len(probs.shape))
        vector_size = probs.shape[1]
        cams = list()
        class_indices = list()

        ### Input image normalization
        imgs = self._normalize_images(imgs)

        for i, prob in enumerate(probs):
            sorted_prob = np.argsort(prob)[::-1]
            img = imgs[i]

            cams_ = list()
            class_indices_ = list()
            for j in range(GradCamPlusPlus.TOP3):
                ### Create one-hot encoding
                class_index = sorted_prob[j]
                one_hot_encoding = self._create_one_hot_encoding(vector_size, class_index)
                class_indices_.append(class_index)

                ### Create CAM image
                cam = self._create_cam_img(sess, img, class_index, one_hot_encoding)
                cam = np.uint8(cam * 255)  # image denormalization
                cams_ = [cam] if len(cams_) == 0 else np.append(cams_, [cam], axis=0)

            cams = [cams_] if len(cams) == 0 else np.append(cams, [cams_], axis=0)
            class_indices.append(class_indices_)

        return cams, class_indices  # cams: (batch size, top 1~3, height, width), class_indices: (batch size, top 1~3)

    def convert_cam_2_heatmap(self, cam):
        # cam color range: 0~255
        assert len(cam.shape) == 2, 'len(cam.shape) == 2, but len(cam.shape): {}'.format(len(cam.shape))

        return cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # rgb

    @staticmethod
    def show_image(img, title='title'):
        cv2.imshow(title, img)
        while True:
            if cv2.waitKey(0) == 113:  # 'q' key
                break

    def overlay_heatmap(self, img, heatmap):
        # heatmap color range: 0~255
        assert len(heatmap.shape) == 3 and heatmap.shape[2] == 3, 'heatmap must be RGB'

        if len(img.shape) != 3 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.min() < 0:
            img = np.maximum(img, 0)
        if img.max() <= 1:
            img = np.uint8(img * 255)

        overay_heatmap = cv2.addWeighted(heatmap, 0.4, img, 0.5, 0)
        # self.show_image(overay_heatmap)
        return overay_heatmap

    def _get_upper_boundary(self, img, height, width):
        assert len(img.shape) == 2, 'len(img.shape) == 2, but len(img.shape): {}'.format(len(img.shape))

        for h in range(height):
            for w in range(width):
                if int(img[h, w]) > GradCamPlusPlus.COLOR_THRESHOLD:
                    return 0 if h == 0 else h - 1

        return None

    def _get_lower_boundary(self, img, height, width):
        assert len(img.shape) == 2, 'len(img.shape) == 2, but len(img.shape): {}'.format(len(img.shape))

        for h in reversed(range(height)):
            for w in range(width):
                if int(img[h, w]) > GradCamPlusPlus.COLOR_THRESHOLD:
                    return height if h == height else h + 1

        return None

    def draw_rectangle(self, img, cam, box_color):
        # cam color range: 0~255
        assert len(box_color) == 3  # rgb
        assert len(cam.shape) == 2, 'len(cam.shape) == 2, but len(cam.shape): {}'.format(len(cam.shape))

        if len(img.shape) != 3 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.min() < 0:
            img = np.maximum(img, 0)
        if img.max() <= 1:
            img = np.uint8(img * 255)

        ### Get height, width
        height, width = cam.shape

        ### Get top, bottom, left, right
        top = self._get_upper_boundary(cam, height, width)
        bottom = self._get_lower_boundary(cam, height, width)
        transpose_cam = np.transpose(cam)
        left = self._get_upper_boundary(transpose_cam, width, height)
        right = self._get_lower_boundary(transpose_cam, width, height)

        if top is None or bottom is None or left is None or right is None:
            return img

        return cv2.rectangle(img, (left, top), (right, bottom), tuple(box_color[::-1]), 3)  # color : bgr!!!!
