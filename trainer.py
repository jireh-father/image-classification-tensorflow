import tensorflow as tf
from model import model_factory
from preprocessing import preprocessing_factory
import optimizer
import os, glob
from datetime import datetime
import dataset
import visualizer
import numpy as np
import json
from tensorflow.contrib.tensorboard.plugins import projector
from grad_cam_plus_plus import GradCamPlusPlus

NUM_DATASET_MAP = {"mnist": [60000, 10000, 10, 1], "cifar10": [50000, 10000, 10, 3], "flowers": [3320, 350, 5, 3],
                   "block": [4579, 510, 3, 1],
                   "direction": [3036, 332, 4, 1]}


def write_summary(writer, name, imgs, sess):
    img_tensor = tf.convert_to_tensor(np.array(imgs))
    image_summaries = tf.summary.image(name, img_tensor, len(imgs))
    merged_image_summary = tf.summary.merge([image_summaries])
    writer.add_summary(sess.run(merged_image_summary))


def train(conf):
    if conf.dataset_name not in NUM_DATASET_MAP:
        num_dataset, num_classes = dataset.make_tfrecord(conf.dataset_name, conf.dataset_dir, conf.train_fraction,
                                                         conf.num_channel,
                                                         conf.num_dataset_parallel)
        if num_dataset is None:
            metadata = json.load(open(os.path.join(conf.dataset_dir, "metadata")))
            NUM_DATASET_MAP[conf.dataset_name] = [metadata["num_train"],
                                                  metadata["num_validation"], metadata["num_classes"],
                                                  conf.num_channel]
        else:
            NUM_DATASET_MAP[conf.dataset_name] = [num_dataset * conf.train_fraction,
                                                  num_dataset * (1 - conf.train_fraction), num_classes,
                                                  conf.num_channel]
    num_channel = NUM_DATASET_MAP[conf.dataset_name][3]
    num_classes = NUM_DATASET_MAP[conf.dataset_name][2]

    is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

    if conf.model_name[:6] == "nasnet":
        model_f = model_factory.get_network_fn(conf.model_name, num_classes, weight_decay=conf.weight_decay,
                                               is_training=is_training)
    else:
        model_f = model_factory.get_network_fn(conf.model_name, num_classes, weight_decay=conf.weight_decay,
                                               is_training=True)

    model_image_size = conf.model_image_size or model_f.default_image_size

    def pre_process(example_proto, training):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
                    'image/width': tf.FixedLenFeature((), tf.int64, default_value=0)
                    }

        parsed_features = tf.parse_single_example(example_proto, features)
        if conf.preprocessing_name:
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(conf.preprocessing_name,
                                                                             is_training=training)
            image = tf.image.decode_image(parsed_features["image/encoded"], num_channel)
            image = tf.clip_by_value(
                image_preprocessing_fn(image, model_image_size, model_image_size), -1, 1.0)
        else:
            image = tf.clip_by_value(tf.image.per_image_standardization(
                tf.image.resize_images(tf.image.decode_jpeg(parsed_features["image/encoded"], num_channel),
                                       [model_image_size, model_image_size])), -1., 1.0)

        if len(parsed_features["image/class/label"].get_shape()) == 0:
            label = tf.one_hot(parsed_features["image/class/label"], num_classes)
        else:
            label = parsed_features["image/class/label"]

        return image, label

    def train_dataset_map(example_proto):
        return pre_process(example_proto, True)

    def test_dataset_map(example_proto):
        return pre_process(example_proto, False)

    def get_model():
        model_name = conf.model_name
        inputs = tf.placeholder(tf.float32, shape=[None, model_image_size, model_image_size, num_channel],
                                name="inputs")

        labels = tf.placeholder(tf.float32, shape=[None, num_classes], name="labels")
        global_step = tf.Variable(0, trainable=False)
        learning_rate = optimizer.configure_learning_rate(NUM_DATASET_MAP[conf.dataset_name][0], global_step, conf)
        # learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        conf.num_channel = num_channel
        conf.num_classes = num_classes
        if model_name in ["deconv", "ed", "deconv_conv"]:
            logits, gen_x, gen_x_ = model_f(inputs, model_conf=conf)
            class_loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            gen_loss_op = tf.log(
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gen_x_, logits=gen_x)))
            loss_op = tf.add(class_loss_op, gen_loss_op)

            ops = [class_loss_op, loss_op, gen_loss_op]
            ops_key = ["class_loss_op", "loss_op", "gen_loss_op"]
        else:
            if model_name == "conv":
                logits = model_f(inputs, model_conf=conf)
            else:
                logits, end_points = model_f(inputs)
            if model_name == "resnet":
                logits = tf.reshape(logits, [-1, num_classes])
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            ops = [loss_op]
            ops_key = ["loss_op"]
        tf.summary.scalar('loss', loss_op)
        opt = optimizer.configure_optimizer(learning_rate, conf)
        train_op = opt.minimize(loss_op, global_step=global_step)
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy_op)
        merged = tf.summary.merge_all()

        return inputs, labels, train_op, accuracy_op, merged, ops, ops_key, logits, end_points

    if not os.path.exists(conf.dataset_dir):
        conf.dataset_dir = os.path.join("/home/data", conf.dataset_name)

    train_filenames = glob.glob(os.path.join(conf.dataset_dir, conf.dataset_name + ("_%s*tfrecord" % conf.train_name)))
    test_filenames = glob.glob(os.path.join(conf.dataset_dir, conf.dataset_name + ("_%s*tfrecord" % conf.test_name)))

    inputs, labels, train_op, accuracy_op, merged, ops, ops_key, logits, end_points = get_model()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    summary_dir = os.path.join(conf.log_dir, "summary")
    train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(summary_dir + '/test')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if conf.restore_model_path and len(glob.glob(conf.restore_model_path + ".data-00000-of-00001")) > 0:
        saver.restore(sess, conf.restore_model_path)

    train_iterator = tf.data.TFRecordDataset(train_filenames).map(train_dataset_map,
                                                                  conf.num_dataset_parallel).shuffle(
        buffer_size=conf.shuffle_buffer).batch(conf.batch_size).make_initializable_iterator()
    train_next = train_iterator.get_next()
    test_iterator = tf.data.TFRecordDataset(test_filenames).map(test_dataset_map, conf.num_dataset_parallel).batch(
        conf.batch_size).make_initializable_iterator()
    test_next = test_iterator.get_next()

    num_train = NUM_DATASET_MAP[conf.dataset_name][0] // conf.batch_size
    num_test = NUM_DATASET_MAP[conf.dataset_name][1] // conf.batch_size
    if conf.vis_epoch is not None:
        config = projector.ProjectorConfig()
        vis_dir = os.path.join(conf.log_dir, "embedding")
        total_dataset = None
        total_labels = None
        total_activations = None
    heatmap_imgs = []
    bb_imgs = []
    for epoch in range(conf.epoch):
        train_step = 0
        if conf.vis_epoch is not None and total_dataset is not None:
            total_dataset = None
            total_labels = None
            total_activations = None
        if conf.train:
            sess.run(train_iterator.initializer)
            total_train_accuracy = .0
            inner_train_step = 0
            while True:
                try:
                    batch_xs, batch_ys = sess.run(train_next)
                    results = sess.run([train_op, merged, accuracy_op, ] + ops,
                                       feed_dict={inputs: batch_xs, labels: batch_ys, is_training: True})
                    total_train_accuracy += results[2]
                    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    if train_step % conf.summary_interval == 0:
                        ops_results = " ".join(list(map(lambda x: str(x), list(zip(ops_key, results[3:])))))
                        print(
                            ("[%s TRAIN %d epoch, %d / %d step] accuracy: %f" % (
                                now, epoch, train_step, num_train, results[2])) + ops_results)
                        train_writer.add_summary(results[1], train_step + epoch * num_train)
                    train_step += 1
                    inner_train_step += 1
                except tf.errors.OutOfRangeError:
                    break
            if inner_train_step > 0:
                print("Avg Train Accuracy : %f" % (float(total_train_accuracy) / inner_train_step))
            if epoch % conf.num_save_interval == 0:
                saver.save(sess, conf.log_dir + "/model_epoch_%d.ckpt" % epoch)
        if conf.eval:
            total_accuracy = 0
            test_step = 0
            sess.run(test_iterator.initializer)

            while True:
                try:
                    test_xs, test_ys = sess.run(test_next)
                    results = sess.run(
                        [merged, accuracy_op, logits] + ops,
                        feed_dict={inputs: test_xs, labels: test_ys, is_training: False})
                    total_accuracy += results[1]
                    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    ops_results = " ".join(list(map(lambda x: str(x), list(zip(ops_key, results[3:])))))
                    print(("[%s TEST %d epoch, %d /%d step] accuracy: %f" % (
                        now, epoch, test_step, num_test, results[1])) + ops_results)
                    test_writer.add_summary(results[0], test_step + (train_step + epoch * num_train))
                    test_step += 1
                    if conf.vis_epoch is not None and epoch % conf.vis_epoch == 0:
                        if conf.num_vis_steps >= test_step:
                            if total_dataset is None:
                                total_dataset = test_xs
                                total_labels = test_ys
                                total_activations = results[2]
                            else:
                                total_dataset = np.append(test_xs, total_dataset, axis=0)
                                total_labels = np.append(test_ys, total_labels, axis=0)
                                total_activations = np.append(results[2], total_activations, axis=0)

                            ### Create CAM image
                            if end_points and conf.num_cam:
                                grad_cam_plus_plus = GradCamPlusPlus(end_points[model_f.default_logit_layer_name],
                                                                     end_points[model_f.default_last_conv_layer_name],
                                                                     inputs)
                                cam_imgs, class_indices = grad_cam_plus_plus.create_cam_imgs(sess, test_xs, results[2])

                                for i in range(conf.num_cam):
                                    box_img = np.copy(test_xs[i])
                                    # for j in range(GradCamPlusPlus.TOP3):
                                    ### Overlay heatmap
                                    heapmap = grad_cam_plus_plus.convert_cam_2_heatmap(cam_imgs[i][0])
                                    overlay_img = grad_cam_plus_plus.overlay_heatmap(test_xs[i], heapmap)
                                    heatmap_imgs.append(overlay_img)

                                    ### Boxing
                                    box_img = grad_cam_plus_plus.draw_rectangle(box_img, cam_imgs[i][0], [255, 0, 0])
                                    bb_imgs.append(box_img)


                except tf.errors.OutOfRangeError:
                    break
            if conf.vis_epoch is not None and epoch % conf.vis_epoch == 0:
                write_summary(test_writer, "heatmap_epoch_" + str(epoch), heatmap_imgs, sess)
                write_summary(test_writer, "bb_epoch_" + str(epoch), bb_imgs, sess)
                bb_imgs = []
                heatmap_imgs = []
            if test_step > 0:
                print("Avg Accuracy : %f" % (float(total_accuracy) / test_step))
                if conf.vis_epoch is not None and epoch % conf.vis_epoch == 0:
                    # vis_dir = os.path.join(conf.log_dir, "embed_vis_%d" % epoch)
                    visualizer.add_embedding(config, sess=sess, embedding_list=[total_activations],
                                             embedding_path=vis_dir, image_size=model_image_size,
                                             channel=num_channel, labels=total_labels, prefix="epoch" + str(epoch))
        if not conf.train:
            break

    ### Write summary
    # write_summary(test_writer, summary_names, result_imgs, sess)

    if conf.vis_epoch is not None:
        visualizer.write_embedding(config, sess, total_dataset, embedding_path=vis_dir, image_size=model_image_size,
                                   channel=num_channel, labels=total_labels)

    sess.close()
