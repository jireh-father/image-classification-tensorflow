def train():
    IMAGE_SIZE_MAP = {"alexnet": 224, "inception": 299, "inception_resnet": 299, "resnet": 224, "conv": 224,
                      "deconv": 224}


    NUM_DATASET_MAP = {"mnist": [60000, 10000], "cifar10": [50000, 10000], "flowers": [3320, 350], "block": [4579, 510],
                       "direction": [3036, 332]}
    is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

    if FLAGS.model_image_size is not None:
        model_image_size = FLAGS.model_image_size
    else:
        if FLAGS.model_name not in IMAGE_SIZE_MAP.keys():
            sys.exit("invalid model name")
        model_image_size = IMAGE_SIZE_MAP[FLAGS.model_name]


    def pre_process(example_proto, training):
        features = {"image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
                    "image/class/label": tf.FixedLenFeature((), tf.int64, default_value=0)}

        parsed_features = tf.parse_single_example(example_proto, features)
        if FLAGS.preprocessing_name:
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocessing_name,
                                                                             is_training=training)
            image = tf.image.decode_image(parsed_features["image/encoded"], FLAGS.num_channel)
            image = tf.clip_by_value(image_preprocessing_fn(image, model_image_size, model_image_size), .0, 1.0)
        else:
            image = tf.clip_by_value(tf.image.per_image_standardization(
                tf.image.resize_images(tf.image.decode_jpeg(parsed_features["image/encoded"], FLAGS.num_channel),
                                       [model_image_size, model_image_size])), .0, 1.0)

        if len(parsed_features["image/class/label"].get_shape()) == 0:
            label = tf.one_hot(parsed_features["image/class/label"], FLAGS.num_classes)
        else:
            label = parsed_features["image/class/label"]

        return image, label


    def train_dataset_map(example_proto):
        return pre_process(example_proto, True)


    def test_dataset_map(example_proto):
        return pre_process(example_proto, False)


    def get_model():
        num_classes = FLAGS.num_classes
        model_name = FLAGS.model_name

        inputs = tf.placeholder(tf.float32, shape=[None, model_image_size, model_image_size, FLAGS.num_channel],
                                name="inputs")

        labels = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes], name="labels")
        global_step = tf.Variable(0, trainable=False)
        learning_rate = optimizer.configure_learning_rate(NUM_DATASET_MAP[FLAGS.dataset_name][0], global_step, FLAGS)
        # learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        if model_name == "deconv":
            logits, gen_x, gen_x_ = deconv.build_model(inputs, num_classes, is_training, FLAGS)
            class_loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            gen_loss_op = tf.log(
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gen_x_, logits=gen_x)))
            loss_op = tf.add(class_loss_op, gen_loss_op)

            ops = [class_loss_op, loss_op, gen_loss_op]
            ops_key = ["class_loss_op", "loss_op", "gen_loss_op"]
        else:
            if model_name == "alexnet":
                logits, end_points = alexnet.alexnet_v2(inputs, num_classes, is_training)
            elif model_name == "inception":
                logits, end_points = inception_v4.inception_v4(inputs, num_classes, is_training)
            elif model_name == "inception_resnet":
                logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs, num_classes, is_training)
            elif model_name == "resnet":
                logits, end_points = resnet_v2.resnet_v2_152(inputs, num_classes, is_training)
                logits = tf.reshape(logits, [-1, num_classes])
            elif model_name == "conv":
                logits = conv.build_model(inputs, num_classes, is_training, FLAGS)
            else:
                sys.exit("invalid model name")
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            ops = [loss_op]
            ops_key = ["loss_op"]

        tf.summary.scalar('loss', loss_op)
        opt = optimizer.configure_optimizer(learning_rate, FLAGS)
        train_op = opt.minimize(loss_op, global_step=global_step)
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('accuracy', accuracy_op)
        merged = tf.summary.merge_all()

        return inputs, labels, train_op, accuracy_op, merged, ops, ops_key


    if not os.path.exists(FLAGS.dataset_dir):
        FLAGS.dataset_dir = os.path.join("/home/data", FLAGS.dataset_name)

    train_filenames = glob.glob(os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name + "_train*tfrecord"))
    test_filenames = glob.glob(os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name + "_validation*tfrecord"))

    inputs, labels, train_op, accuracy_op, merged, ops, ops_key = get_model()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if FLAGS.restore_model_path and len(glob.glob(FLAGS.restore_model_path + ".data-00000-of-00001")) > 0:
        saver.restore(sess, FLAGS.restore_model_path)

    train_iterator = tf.data.TFRecordDataset(train_filenames).map(train_dataset_map, FLAGS.num_dataset_parallel).shuffle(
        buffer_size=FLAGS.shuffle_buffer).batch(FLAGS.batch_size).make_initializable_iterator()
    test_iterator = tf.data.TFRecordDataset(test_filenames).map(test_dataset_map, FLAGS.num_dataset_parallel).batch(
        FLAGS.batch_size).make_initializable_iterator()

    train_step = 0
    for epoch in range(FLAGS.epoch):
        if FLAGS.train:
            sess.run(train_iterator.initializer)
            while True:
                try:
                    batch_xs, batch_ys = sess.run(train_iterator.get_next())
                    results = sess.run([train_op, merged, accuracy_op] + ops,
                                       feed_dict={inputs: batch_xs, labels: batch_ys, is_training: True})
                    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    if train_step % FLAGS.summary_interval == 0:
                        ops_results = " ".join(list(map(lambda x: str(x), list(zip(ops_key, results[3:])))))
                        print(
                            ("[%s TRAIN %d epoch, %d step] accuracy: %f" % (
                                now, epoch, train_step, results[2])) + ops_results)
                        train_writer.add_summary(results[1], train_step)
                    train_step += 1
                except tf.errors.OutOfRangeError:
                    break
            saver.save(sess, FLAGS.log_dir + "/model_epoch_%d.ckpt" % epoch)
        if FLAGS.eval:
            total_accuracy = 0
            test_step = train_step
            sess.run(test_iterator.initializer)
            while True:
                try:
                    test_xs, test_ys = sess.run(test_iterator.get_next())
                    results = sess.run(
                        [merged, accuracy_op] + ops, feed_dict={inputs: test_xs, labels: test_ys, is_training: False})
                    total_accuracy += results[1]
                    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    ops_results = " ".join(list(map(lambda x: str(x), list(zip(ops_key, results[2:])))))
                    print(("[%s TEST %d epoch, %d step] accuracy: %f" % (now, epoch, test_step, results[1])) + ops_results)
                    test_writer.add_summary(results[0], test_step)
                    test_step += 1
                except tf.errors.OutOfRangeError:
                    break
            if test_step > 0:
                print("Avg Accuracy : %f" % (float(total_accuracy) / test_step))
        if not FLAGS.train:
            break
