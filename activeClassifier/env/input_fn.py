import tensorflow as tf


def parse_function(filename, label, FLAGS):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=FLAGS.img_shape[-1])

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape(FLAGS.img_shape)

    if FLAGS.dataset == "omniglot":
        # change lines to 1 and empty space to 0 as in mnist to be potentially able to cross-train
        image = tf.abs(image - 1)
    return image, label


def translate_function(image, label, FLAGS):
    '''
    Not sure if translation differs every epoch or not atm.
    Alternative: could pass a vector with pre-sampled x1, y1 (and a counter to index) along to ensure same translation.
    '''
    if FLAGS.translated_size:
        pad_height = FLAGS.translated_size - FLAGS.img_shape[0]
        pad_width = FLAGS.translated_size - FLAGS.img_shape[1]

        image = tf.reshape(image, FLAGS.img_shape)

        y1 = tf.random_uniform(shape=[], maxval=pad_height, dtype=tf.int32)
        x1 = tf.random_uniform(shape=[], maxval=pad_width, dtype=tf.int32)
        image = tf.pad(image, [(y1, pad_height - y1), (x1, pad_width - x1), (0,0)], mode='constant', constant_values=0.)

    return image, label


def resize_function(image, label, FLAGS):
    image = tf.image.resize_images(image, 2 * [FLAGS.img_resize], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, label


def pipeline(data, batch_sz, FLAGS, shuffle, repeats, n, preftch=4):
    translate_fn = lambda img, label: translate_function(img, label, FLAGS)
    parse_fn = lambda img, label: parse_function(img, label, FLAGS)
    resize_fn = lambda img, label: resize_function(img, label, FLAGS)

    out_data = (tf.data.Dataset.from_tensor_slices(data)
                .shuffle(buffer_size=tf.cast(n, tf.int64), reshuffle_each_iteration=shuffle)
                )
    if FLAGS.dataset in ["MNIST_cluttered", "omniglot"]:
        out_data = out_data.map(parse_fn, num_parallel_calls=FLAGS.num_parallel_preprocess)
    if FLAGS.img_resize:
        out_data = out_data.map(resize_fn, num_parallel_calls=FLAGS.num_parallel_preprocess)
    if FLAGS.translated_size:
        out_data = out_data.map(translate_fn, num_parallel_calls=FLAGS.num_parallel_preprocess)
    if FLAGS.cache:
        out_data = out_data.cache()

    out_data = (out_data
                # .apply(batch_and_drop_remainder(FLAGS.batch_size))  # tSNE requires known batch_sz
                .batch(batch_sz)
                .repeat(repeats)
                .prefetch(preftch)
                )

    return out_data


def input_fn(FLAGS, batch_sz):
    '''train, valid, test: tuples of (images, labels)'''
    def mask_batch_sz(shape):
        return [None] + list(shape[1:])

    features_ph_train = tf.placeholder(FLAGS.data_dtype[0], mask_batch_sz(FLAGS.train_data_shape[0]))
    labels_ph_train   = tf.placeholder(FLAGS.data_dtype[1], mask_batch_sz(FLAGS.train_data_shape[1]))
    features_ph_valid = tf.placeholder(FLAGS.data_dtype[0], mask_batch_sz(FLAGS.valid_data_shape[0]))
    labels_ph_valid   = tf.placeholder(FLAGS.data_dtype[1], mask_batch_sz(FLAGS.valid_data_shape[1]))
    features_ph_test  = tf.placeholder(FLAGS.data_dtype[0], mask_batch_sz(FLAGS.test_data_shape[0]))
    labels_ph_test    = tf.placeholder(FLAGS.data_dtype[1], mask_batch_sz(FLAGS.test_data_shape[1]))

    tr_data    = pipeline((features_ph_train, labels_ph_train), batch_sz, FLAGS, repeats=tf.cast(tf.ceil(FLAGS.num_epochs + FLAGS.num_epochs / FLAGS.eval_step_interval), tf.int64), shuffle=True, n=FLAGS.train_data_shape[0][0])
    # repeats * 2 because also used for visualization etc.
    valid_data = pipeline((features_ph_valid, labels_ph_valid), batch_sz, FLAGS, repeats=tf.cast(tf.ceil(20 *  FLAGS.num_epochs / FLAGS.eval_step_interval), tf.int64), shuffle=False, n=FLAGS.valid_data_shape[0][0])
    test_data  = pipeline((features_ph_test, labels_ph_test), batch_sz, FLAGS, repeats=tf.cast(1 + tf.ceil(FLAGS.num_epochs / FLAGS.eval_step_interval), tf.int64), shuffle=False, n=FLAGS.test_data_shape[0][0])
    if FLAGS.img_resize:
        FLAGS.img_shape[0:2] = 2 * [FLAGS.img_resize]
    if FLAGS.translated_size:
        FLAGS.img_shape[0:2] = 2 * [FLAGS.translated_size]

    handle = tf.placeholder(tf.string, shape=[], name='handle')
    iterator = tf.data.Iterator.from_string_handle(handle, tr_data.output_types, tr_data.output_shapes)
    # iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    images, labels = iterator.get_next()

    train_init_op = tr_data.make_initializable_iterator()
    valid_init_op = valid_data.make_initializable_iterator()
    test_init_op  = test_data.make_initializable_iterator()

    # train_init_op = iterator.make_initializer(tr_data)
    # valid_init_op = iterator.make_initializer(valid_data)
    # test_init_op  = iterator.make_initializer(test_data)

    inputs = {'images': images,
              'labels': labels,
              'features_ph_train': features_ph_train,
              'labels_ph_train'  : labels_ph_train,
              'features_ph_valid': features_ph_valid,
              'labels_ph_valid'  : labels_ph_valid,
              'features_ph_test' : features_ph_test,
              'labels_ph_test'   : labels_ph_test,
              'handle': handle,
              'train_init_op': train_init_op,
              'valid_init_op': valid_init_op,
              'test_init_op':  test_init_op}
    return inputs
