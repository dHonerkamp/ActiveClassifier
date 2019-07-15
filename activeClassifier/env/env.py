import numpy as np
import tensorflow as tf
import logging
logger = logging.getLogger(__name__)

from activeClassifier.env.input_fn import input_fn


class ImageForagingEnvironment:
    def __init__(self, FLAGS, name='env'):
        self.name = name
        with tf.variable_scope(self.name, reuse=False):
            # setup data
            self.batch_size_orig = tf.placeholder_with_default(FLAGS.batch_size, shape=(), name='batch_size')
            self.inputs = input_fn(FLAGS, tf.cast(self.batch_size_orig, tf.int64))

            self.MC_samples = tf.placeholder(tf.int32, shape=(), name='MC_samples')
            self.x = tf.placeholder_with_default(self.inputs['images'], shape=[None] + FLAGS.img_shape, name='x')
            self.y = tf.cast(tf.placeholder_with_default(self.inputs['labels'], shape=[None], name='y'), tf.int32)

            self.x_MC     = tf.tile(self.x, tf.stack([self.MC_samples, 1, 1, 1]))
            self.y_MC     = tf.tile(self.y, tf.expand_dims(self.MC_samples, axis=0))
            self.B = tf.shape(self.x_MC)[0]  # potentially variable batch_size
            self.img_NHWC = tf.reshape(self.x_MC, [self.B] + FLAGS.img_shape)

            # setup glimpse extraction
            self.img_shape  = FLAGS.img_shape
            self.scales     = FLAGS.scale_sizes
            self.num_scales = len(self.scales)
            self.padding    = FLAGS.padding
            self.patch_shape = [self.num_scales, FLAGS.scale_sizes[0], FLAGS.scale_sizes[0], FLAGS.img_shape[-1]]
            self.patch_shape_flat = np.prod(self.patch_shape)
            if FLAGS.resize_method == "AVG":
                self.resize_method = lambda glimpse, ratio: tf.nn.pool(glimpse,
                                                                       window_shape=[ratio, ratio],
                                                                       strides=[ratio, ratio],
                                                                       pooling_type="AVG",
                                                                       padding="SAME")
            elif FLAGS.resize_method == "BILINEAR":
                self.resize_method = lambda glimpse, ratio: tf.image.resize_images(glimpse,
                                                                                   [self.scales[0], self.scales[0]],
                                                                                   method=tf.image.ResizeMethod.BILINEAR)
            elif FLAGS.resize_method == "BICUBIC":
                self.resize_method = lambda glimpse, ratio: tf.image.resize_images(glimpse,
                                                                                   [self.scales[0], self.scales[0]],
                                                                                   method=tf.image.ResizeMethod.BICUBIC)
    def intialise(self, train_data, valid_data, test_data, sess):
        features_ph_train = self.inputs['features_ph_train']
        labels_ph_train = self.inputs['labels_ph_train']
        features_ph_valid = self.inputs['features_ph_valid']
        labels_ph_valid = self.inputs['labels_ph_valid']
        features_ph_test = self.inputs['features_ph_test']
        labels_ph_test = self.inputs['labels_ph_test']

        self.handle = self.inputs['handle']

        self.train_init_op = self.inputs['train_init_op']
        self.valid_init_op = self.inputs['valid_init_op']
        self.test_init_op = self.inputs['test_init_op']

        train_handle = sess.run(self.train_init_op.string_handle())
        valid_handle = sess.run(self.valid_init_op.string_handle())
        test_handle = sess.run(self.test_init_op.string_handle())
        sess.run(self.train_init_op.initializer, feed_dict={features_ph_train: train_data[0],
                                                            labels_ph_train  : train_data[1]})
        sess.run(self.valid_init_op.initializer, feed_dict={features_ph_valid: valid_data[0],
                                                            labels_ph_valid  : valid_data[1]})
        sess.run(self.test_init_op.initializer, feed_dict={features_ph_test: test_data[0],
                                                           labels_ph_test  : test_data[1]})

        return {'train': train_handle, 'valid': valid_handle, 'test': test_handle}

    def step(self, loc, decision):
        with tf.variable_scope(self.name):
            if self.padding == "zero":
                img_NHWC_padded, adj_loc = self._extract_glimpse_zero_padding_fix(self.img_NHWC, self.scales[-1], self.scales[-1], loc)
                glimpse = tf.image.extract_glimpse(img_NHWC_padded, [self.scales[-1], self.scales[-1]], adj_loc)
            else:
                glimpse = tf.image.extract_glimpse(self.img_NHWC, [self.scales[-1], self.scales[-1]], loc, uniform_noise=(self.padding == "uniform"))

            # indices of where the glimpse pixel are coming from
            img_idx = tf.stack(tf.meshgrid(tf.range(self.B), *[tf.range(s) for s in self.img_NHWC.shape[1:3]], indexing='ij'), axis=-1)
            if self.scales != [8]:
                logger.warning('glimpse_idx might be incorrect for these scales')
                # TODO: WHAT TO DO WITH THE LOCATIONS GOING OVER THE EDGE?? FOR NOW, CHEAT A LITTLE AND CLIP LOSC TO 0.9 INSTEAD OF 1. WON'T WORK FOR ALL SCALE / IMG SIZES!
                # TODO: NOT HANDLING MULTIPLE SCALES
            glimpse_idx = tf.image.extract_glimpse(tf.cast(img_idx, tf.float32), [self.scales[-1], self.scales[-1]], tf.clip_by_value(loc, -0.9, 0.9))
            glimpse_idx = tf.cast(glimpse_idx, tf.int32)

            if self.num_scales == 1:
                next_glimpse = tf.layers.flatten(glimpse)
            else:
                next_glimpse = self._multi_scale_glimpse(glimpse)

            corr_classification, done = self._process_decision(decision)
            next_glimpse = tf.where(done, tf.zeros_like(next_glimpse), next_glimpse)

            return next_glimpse, glimpse_idx, corr_classification, done

    def _process_decision(self, decision):
        corr_classification = tf.cast(tf.equal(decision, self.y_MC), tf.float32)
        # non-classification action is encoded as -1
        done = tf.not_equal(decision, -1)
        return corr_classification, done

    def _multi_scale_glimpse(self, glimpse):
        # tf.while should allow to process scales in parallel. Small improvement
        ta = tf.TensorArray(dtype=tf.float32, size=self.num_scales, infer_shape=True, dynamic_size=False)

        def _add_patch(i, ta):
            sc = tf.gather(self.scales, i)
            start_end = (self.scales[-1] - sc) // 2
            patch = glimpse[:, start_end:self.scales[-1] - start_end,
                    start_end:self.scales[-1] - start_end, :]
            ratio = sc // self.scales[0]
            patch = self.resize_method(patch, ratio)

            ta = ta.write(i, patch)
            i += 1
            return i, ta

        final_i, final_ta = tf.while_loop(
                cond=lambda i, _: tf.less(i, self.num_scales),
                body=_add_patch,
                loop_vars=[tf.constant(0), ta],
        )

        # [batch_sz, num_scales, H, W, C]
        patches = tf.transpose(final_ta.stack(), [1, 0, 2, 3, 4])
        patches_flat = tf.layers.flatten(patches)
        blub = self.patch_shape_flat
        patches_flat.set_shape([None, blub])

        return patches_flat

    def _extract_glimpse_zero_padding_fix(self, img_batch, max_glimpse_width, max_glimpse_heigt, offset):
        with tf.name_scope('extract_glimpse_zero_padding_fix'):
            orig_sz = tf.constant(img_batch.get_shape().as_list()[1:3])
            padded_sz = orig_sz + tf.stack([max_glimpse_heigt, max_glimpse_width])

            img_batch_padded = tf.pad(img_batch,
                                      [(0, 0), (max_glimpse_heigt // 2, max_glimpse_heigt // 2),
                                       (max_glimpse_width // 2, max_glimpse_width // 2), (0, 0)])

            new_offset = offset * tf.cast(orig_sz, dtype=tf.float32) / tf.cast(padded_sz, tf.float32)

        return img_batch_padded, new_offset
    
    def composed_glimpse(self, FLAGS, glimpse, num_glimpses_dyn):
        """
        Args:
            glimpse: [T,B,pixels]
        """
        self.glimpses_composed = []
        downscaled_scales = []
        num_glimpses_p0 = FLAGS.num_glimpses
        num_scales = len(FLAGS.scale_sizes)
        scale0 = FLAGS.scale_sizes[0]
        out_sz = FLAGS.scale_sizes[-1]
        C = FLAGS.img_shape[-1]

        # padding up to max_num_glimpses, as tf.split can't except varying number of splits
        t_diff = FLAGS.num_glimpses - num_glimpses_dyn
        glimpse_padded = tf.pad(glimpse, [(0, t_diff), (0, 0), (0, 0)])

        masks, paddings = [], []
        for idx in range(num_scales):
            pad_size = (out_sz - FLAGS.scale_sizes[idx]) // 2
            padding = tf.constant([[0, 0],
                                   [pad_size, out_sz - FLAGS.scale_sizes[idx] - pad_size],
                                   [pad_size, out_sz - FLAGS.scale_sizes[idx] - pad_size],
                                   [0, 0]])

            mask = tf.ones([self.B * num_glimpses_p0, FLAGS.scale_sizes[idx], FLAGS.scale_sizes[idx], C])
            mask = tf.pad(mask, padding, mode='CONSTANT', constant_values=0)

            masks.append(mask)
            paddings.append(padding)

        glimpses_reshpd = tf.reshape(glimpse_padded, [self.B * num_glimpses_p0, -1])
        glimpse_composed = tf.zeros([self.B * num_glimpses_p0, out_sz, out_sz, C], tf.float32)
        scales = tf.split(glimpses_reshpd, num_scales, axis=1)
        last_mask = tf.zeros([self.B * num_glimpses_p0, out_sz, out_sz, C])

        # to check actual model env. Nesting from out to in: scales, glimpses, batch
        for idx in range(num_scales):
            downscaled_scales.append(tf.split(
                    tf.reshape(scales[idx], [self.B * num_glimpses_p0, scale0, scale0, C]),
                    num_glimpses_p0, axis=0))

        # Start with smallest scale, pad up to largest, multiply by (mask - last_mask) indicating area not covered by smaller masks
        for idx in range(num_scales):
            scales[idx] = tf.reshape(scales[idx], [self.B * num_glimpses_p0, scale0, scale0, C])  # resize_images expects [B,H,W,C] -> add channel for MNIST

            # repeat and tile glimpse to scale size (unfortunately there is no tf.repeat)
            repeats = FLAGS.scale_sizes[idx] // scale0
            scales[idx] = tf.transpose(scales[idx], [0, 3, 1, 2])  # put channels in front

            scales[idx] = tf.reshape(
                    tf.tile(tf.reshape(scales[idx], [self.B * num_glimpses_p0, C, scale0 ** 2, 1]),
                            [1, 1, 1, repeats]),
                    [self.B * num_glimpses_p0, C, scale0, repeats * scale0])
            scales[idx] = tf.reshape(
                    tf.tile(tf.reshape(tf.transpose(scales[idx], [0, 1, 3, 2]),
                                       [self.B * num_glimpses_p0, C, repeats * scale0 ** 2, 1]),
                            [1, 1, 1, repeats]),
                    [self.B * num_glimpses_p0, C, repeats * scale0, repeats * scale0])

            scales[idx] = tf.transpose(scales[idx], [0, 3, 2, 1])  # put channels back

            # alternative, but not identical to what model actually sees:
            # scales[idx] = tf.image.resize_images(scales[idx], 2*[FLAGS.scale_sizes[idx]], method=tf.image.ResizeMethod.BILINEAR)

            glimpse_composed += (masks[idx] - last_mask) * tf.pad(scales[idx], paddings[idx], mode='CONSTANT',
                                                                  constant_values=0.)
            last_mask = masks[idx]
        # return tf.split(glimpse_composed, num_glimpses_p0, axis=0)
        return tf.reshape(glimpse_composed, [num_glimpses_p0, self.B, out_sz, out_sz, C])
