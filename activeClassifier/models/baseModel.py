import numpy as np
import tensorflow as tf
import logging
logger = logging.getLogger(__name__)

from activeClassifier.tools.tf_tools import TINY


class BaseModel:
    def __init__(self, FLAGS, env, phase, name):
        self.name = name
        self.global_step = tf.train.create_global_step()
        self.global_epoch = self.global_step // FLAGS.train_batches_per_epoch
        self.total_steps = FLAGS.num_epochs * FLAGS.train_batches_per_epoch
        self.epoch_num = self.global_step // FLAGS.train_batches_per_epoch
        self.debug = FLAGS.debug

        with tf.name_scope('Placeholder'):
            self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        with tf.name_scope('Learning_rate'):
            self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step,
                                                            decay_steps=FLAGS.train_batches_per_epoch,
                                                            decay_rate=FLAGS.learning_rate_decay_factor)
            self.learning_rate = tf.maximum(self.learning_rate, FLAGS.min_learning_rate)

        # train phases
        self.phase = tf.convert_to_tensor(phase['name'])

        # references to env variables to simplify feed evaluation
        self.MC_samples = env.MC_samples
        self.handle = env.handle
        self.x_MC = env.x_MC
        self.y_MC = env.y_MC
        self.B = env.B

        # prior preferences
        prior_preferences_glimpse = np.full([FLAGS.num_glimpses, 1], 0., dtype=np.float32)
        prior_preferences_glimpse[4:] = FLAGS.prior_preference_glimpses  # discourage taking more glimpses than neccessary. Visual foraging: -2 * FLAGS.prior_preference_c
        prior_preferences_classification = np.tile(np.array([FLAGS.prior_preference_c, -2. * FLAGS.prior_preference_c], ndmin=2, dtype=np.float32),
                                                   [FLAGS.num_glimpses, 1])
        prior_preferences = np.concatenate(([prior_preferences_glimpse, prior_preferences_classification]), axis=1)
        self.C = tf.nn.log_softmax(prior_preferences)  # [T, [prediction error, correct classification, wrong classification]]

    def _create_train_op(self, FLAGS, loss, global_step, name, varlist=None):
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = train_op.compute_gradients(loss, var_list=varlist)

        # check none gradients
        [logger.warning('NONE gradient: {:}, {}'.format(g, v)) for g, v in grads_and_vars if g is None]

        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            # grad = tf.Print(grad, [var.name, tf.reduce_any(tf.is_nan(grad))])
            clipped_grads_and_vars.append((tf.clip_by_norm(grad, FLAGS.max_gradient_norm), var))

        train_op = train_op.apply_gradients(clipped_grads_and_vars, global_step=global_step, name=name)

        if FLAGS.debug:
            ctrls = [tf.check_numerics(grad, message='grad_check, var: {}, grad: {}'.format(var, grad.name)) for grad, var in grads_and_vars]
            train_op = tf.group([train_op] + ctrls)

        return train_op, grads_and_vars

    def _write_zero_out(self, time, ta, candidate, done, name):
        if self.debug and (candidate.dtype == tf.float32):
            ctrl = [tf.logical_not(tf.reduce_any(tf.is_nan(candidate)), name='ctrl_{}'.format(name))]
        else:
            ctrl = []
        with tf.control_dependencies(ctrl):
            ta = ta.write(time, tf.where(done, tf.zeros_like(candidate), candidate))
        return ta

    @staticmethod
    def _create_ta(records):
        """records: [(name, type)]"""
        return {name: tf.TensorArray(type, size=size, dynamic_size=False, name=name) for name, type, size in records}

    def _known_unknown_accuracy(self, FLAGS, classification):
        if FLAGS.uk_label:
            corr = tf.equal(self.y_MC, classification)
            is_uk = tf.equal(self.y_MC, FLAGS.uk_label)
            corr_kn, corr_uk = tf.dynamic_partition(corr, partitions=tf.cast(is_uk, tf.int32), num_partitions=2)
            acc_kn = tf.reduce_mean(tf.cast(corr_kn, tf.float32))
            acc_uk = tf.reduce_mean(tf.cast(corr_uk, tf.float32))  # can be nan if there are no uks
            share_clf_uk = tf.reduce_mean(tf.cast(tf.equal(classification, FLAGS.uk_label), tf.float32))
        else:
            acc_kn, acc_uk, share_clf_uk = tf.constant(0.), tf.constant(0.), tf.constant(0.)
        return acc_kn, acc_uk, share_clf_uk

    def _create_saver(self, phase):
        return tf.train.Saver(tf.global_variables(), max_to_keep=1, name='Saver_' + phase['name'])  # tf.trainable_variables() for smaller cp


    def get_feeds(self, FLAGS, handles):
        train_feed = {self.is_training: True,
                      self.handle     : handles['train'],
                      self.MC_samples : FLAGS.MC_samples}

        eval_feed_train = {self.is_training: True,
                           self.handle     : handles['valid'],
                           self.MC_samples : FLAGS.MC_samples}
        eval_feed_valid = {self.is_training: False,
                           self.handle     : handles['valid'],
                           self.MC_samples : FLAGS.MC_samples}
        eval_feed_test  = {self.is_training: False,
                           self.handle     : handles['test'],
                           self.MC_samples : FLAGS.MC_samples}

        return {'train': train_feed, 'eval_train': eval_feed_train, 'eval_valid': eval_feed_valid, 'eval_test': eval_feed_test}


    def get_train_op(self, FLAGS):
        raise NotImplementedError("Abstract method")

    def get_visualisation_fetch(self):
        raise NotImplementedError("Abstract method")
