import shutil
import numpy as np
import tensorflow as tf
import logging
from parameterized import parameterized

from activeClassifier.tools.utility import Utility
from activeClassifier.env.get_data import get_data, random_uk_selection
from activeClassifier.env.env import ImageForagingEnvironment
from activeClassifier.phase_config import get_phases
from activeClassifier.training import run_phase

from activeClassifier.modules.planner.base import Base


class ObservationAveragingTest(tf.test.TestCase):
    def test_repeat_axis(self):
        c_believes = tf.constant(np.array([0.5, 0.5])[np.newaxis, :])
        exp_obs = tf.constant(np.array([1., 5.])[np.newaxis, np.newaxis, :, np.newaxis])

        avg = tf.einsum('bh,bkhg->bkg', c_believes, exp_obs)
        expected = np.array([3.])[np.newaxis, np.newaxis, :]

        with self.test_session():
            self.assertAllEqual(expected, avg.eval())


class ActInfPlannerTest(tf.test.TestCase):
    def test_hyp_tiling(self):
        B = 2
        n_pol = 1
        hyp = Base._hyp_tiling(n_classes=5, n_tiles=B*n_pol)
        with self.test_session():
            self.assertAllEqual(hyp.eval(), [[1., 0., 0., 0., 0.],
                                             [0., 1., 0., 0., 0.],
                                             [0., 0., 1., 0., 0.],
                                             [0., 0., 0., 1., 0.],
                                             [0., 0., 0., 0., 1.],
                                             [1., 0., 0., 0., 0.],
                                             [0., 1., 0., 0., 0.],
                                             [0., 0., 1., 0., 0.],
                                             [0., 0., 0., 1., 0.],
                                             [0., 0., 0., 0., 1.]])


class EnvTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.FLAGS, cls.config = Utility.init(experiment_name='TEST')

        # cut down runtime
        cls.FLAGS.num_epochs = 2
        cls.FLAGS.batch_size = 2
        cls.FLAGS.MC_samples = 1
        cls.FLAGS.num_glimpses = 3
        cls.FLAGS.debug = 1

        if cls.FLAGS.uk_folds:
            n_classes_orig = cls.FLAGS.num_alphabets if cls.FLAGS.dataset == 'omniglot' else cls.FLAGS.num_classes
            cls.FLAGS = random_uk_selection(cls.FLAGS, n_classes_orig)

        # load datasets
        cls.train_data, cls.valid_data, cls.test_data = get_data(cls.FLAGS)

        # cut down runtime
        cls.FLAGS.train_batches_per_epoch = 11
        cls.FLAGS.batches_per_eval_valid = 5
        cls.FLAGS.batches_per_eval_test = 5

    @classmethod
    def tearDownClass(cls):
        logging.shutdown()
        shutil.rmtree(cls.FLAGS.path, ignore_errors=True)

    def test_glimpse_idx(self):
        B = 2
        loc = [[-1., -1.],
               [0., 0.]]
        decision = tf.fill([B], -1)

        env = ImageForagingEnvironment(self.FLAGS)
        next_glimpse, glimpse_idx, corr_classification, done = env.step(loc, decision)

        with tf.Session() as sess:
            handles = env.intialise(self.train_data, self.valid_data, self.test_data, sess)

            out = sess.run(glimpse_idx, feed_dict={env.handle: handles['train'],
                                                   env.MC_samples: 1})

            self.assertEqual(out.shape, (2, 8, 8, 3))
            # random noise padding means indices for places to go across the edge will be messed up (out[0])
            print(out[0])

            # result depends on img and scale size
            assert self.FLAGS.img_shape == [28, 28, 1]
            assert self.FLAGS.scale_sizes == [8]

            # out[0] should be the indices around the top left corner of the image
            # will only use the middle indices, as it'll be downsized through convolutions
            self.assertAllEqual(out[0, 2:6, 2:6], [[[0, 0, 0],
                                                    [0, 0, 1],
                                                    [0, 0, 2],
                                                    [0, 0, 3],],
                                                   [[0, 1, 0],
                                                    [0, 1, 1],
                                                    [0, 1, 2],
                                                    [0, 1, 3],],
                                                   [[0, 2, 0],
                                                    [0, 2, 1],
                                                    [0, 2, 2],
                                                    [0, 2, 3],],
                                                   [[0, 3, 0],
                                                    [0, 3, 1],
                                                    [0, 3, 2],
                                                    [0, 3, 3],]])

            # out[1] should be the indices around the center of the image
            self.assertAllEqual(out[1, 2:6, 2:6], [[[1, 12, 12],
                                                     [1, 12, 13],
                                                     [1, 12, 14],
                                                     [1, 12, 15],],
                                                    [[1, 13, 12],
                                                     [1, 13, 13],
                                                     [1, 13, 14],
                                                     [1, 13, 15],],
                                                    [[1, 14, 12],
                                                     [1, 14, 13],
                                                     [1, 14, 14],
                                                     [1, 14, 15],],
                                                    [[1, 15, 12],
                                                     [1, 15, 13],
                                                     [1, 15, 14],
                                                     [1, 15, 15],]])


class ModelIntegrationTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.FLAGS, cls.config = Utility.init(experiment_name='TEST')

        # cut down runtime
        cls.FLAGS.num_epochs = 2
        cls.FLAGS.batch_size = 2
        cls.FLAGS.MC_samples = 1
        cls.FLAGS.num_glimpses = 3
        cls.FLAGS.debug = 0

        if cls.FLAGS.uk_folds:
            n_classes_orig = cls.FLAGS.num_alphabets if cls.FLAGS.dataset == 'omniglot' else cls.FLAGS.num_classes
            cls.FLAGS = random_uk_selection(cls.FLAGS, n_classes_orig)

        # load datasets
        cls.train_data, cls.valid_data, cls.test_data = get_data(cls.FLAGS)

        # cut down runtime
        cls.FLAGS.train_batches_per_epoch = 11
        cls.FLAGS.batches_per_eval_valid = 5
        cls.FLAGS.batches_per_eval_test = 5

    @classmethod
    def tearDownClass(cls):
        logging.shutdown()
        shutil.rmtree(cls.FLAGS.path, ignore_errors=True)

    @parameterized.expand([
        ['AI_clf', 'ActInf', 'clf'],
        ['AI_G', 'ActInf', 'G'],
        ['AI_G1', 'ActInf', 'G1'],
        ['RL_clf', 'RL', 'clf'],
    ])
    def test_planner(self, name, planner, rl_reward):
        self.FLAGS.planner = planner
        self.FLAGS.rl_reward = rl_reward

        phases = get_phases(self.FLAGS)

        writers = Utility.init_writers(self.FLAGS)
        initial_phase = True

        for phase in phases:
            if phase['num_epochs'] > 0:
                run_phase(self.FLAGS, phase, initial_phase, self.config, writers, self.train_data, self.valid_data, self.test_data)
                initial_phase = False

        for writer in writers.values():
            writer.close()


if __name__ == '__main__':
    tf.test.main()
