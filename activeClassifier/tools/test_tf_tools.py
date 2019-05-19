import tensorflow as tf

from tools.tf_tools import binary_entropy


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


if __name__ == '__main__':
    tf.test.main()