import numpy as np
import tensorflow as tf

from tools.tf_tools import binary_entropy, repeat_axis


class EntropyTest(tf.test.TestCase):
    def test_binary_entropy_logits(self):
        H1 = binary_entropy(logits=[0., 0.])  # i.e. sigmoid(logits) = 0.5
        H0 = binary_entropy(logits=[100., -100.])

        with self.test_session():
            self.assertAllEqual(H1.eval(), [1., 1.])
            self.assertAllClose(H0.eval(), [0., 0.])

    def test_binary_entropy_probs(self):
        H1 = binary_entropy(probs=tf.constant([0.5, 0.5]))
        H0 = binary_entropy(probs=tf.constant([0., 1.]))

        with self.test_session():
            self.assertAllEqual(H1.eval(), [1., 1.])
            self.assertAllEqual(H0.eval(), [0., 0.])


class RepeatsTest(tf.test.TestCase):
    def test_repeat_axis(self):
        x = np.random.rand(10, 10)

        x1 = np.repeat(x, repeats=3, axis=1)
        x2 = repeat_axis(tf.constant(x), axis=1, repeats=5)

        with self.test_session():
            self.assertAllEqual(x1, x2.eval())

if __name__ == '__main__':
    tf.test.main()