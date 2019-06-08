import numpy as np
import tensorflow as tf

from tools.tf_tools import TINY


class Base:
    def __init__(self, FLAGS, env, phase):
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
        self.B = env.batch_sz

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
        [print('{:}, {}'.format(g, v)) for g, v in grads_and_vars if g is None]

        # if global_step is not None:
        #     clipped_grads_and_vars = []
        #     for grad, var in grads_and_vars:
        #         grad = tf.check_numerics(grad, 'nan_' + var.name)
        #         clipped_grads_and_vars.append((tf.clip_by_norm(grad, FLAGS.max_gradient_norm), var))
        # else:
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.max_gradient_norm), var) for grad, var in grads_and_vars]
        train_op = train_op.apply_gradients(clipped_grads_and_vars, global_step=global_step, name=name)
        gradient_check = {v: tf.reduce_mean(g) for g, v in clipped_grads_and_vars}

        return train_op, gradient_check, grads_and_vars

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
        return None
