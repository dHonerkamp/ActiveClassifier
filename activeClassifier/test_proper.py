import shutil
import numpy as np
import tensorflow as tf
import logging
from parameterized import parameterized

from tools.utility import Utility
from env.get_data import get_data, random_uk_selection
from phase_config import get_phases
from training import run_phase

from modules.planner.base import Base


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


class ModelIntegrationTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.FLAGS, cls.config = Utility.init(experiment_name='TEST')

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
