import numpy as np
import tensorflow as tf

from modules.planner.ActInfPlanner import ActInfPlanner


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
        hyp = ActInfPlanner._hyp_tiling(n_classes=5, n_tiles=B*n_pol)
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


if __name__ == '__main__':
    tf.test.main()
