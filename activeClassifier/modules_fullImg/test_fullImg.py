import shutil
import numpy as np
import tensorflow as tf
import logging
from parameterized import parameterized

from activeClassifier.modules_fullImg.planner import maximum_patch


class MaximumPatchTest(tf.test.TestCase):
    def test_maximum_patch(self):
        glimpse_shp = [3, 3]
        img = np.array([[[0, 1, 2],
                         [3, 4, 5],
                         [6, 7, 8]],
                        [[-5, -5, -5],
                         [3, 4, -5],
                         [6, 7, -5]]
                        ], dtype=np.float32)
        img = img[:, :, :, np.newaxis]

        # img = np.tile(img, [2, 1, 1, 1])

        max_patch = maximum_patch(tf.constant(img), glimpse_shp)
        max_patch = tf.squeeze(max_patch, axis=3)
        # unravel_index returns a tuple of (row_idx, col_idx), but as a single tensor
        row_col_tuple = tf.unravel_index(tf.argmax(tf.layers.flatten(max_patch), axis=-1), glimpse_shp)
        idx = tf.transpose(row_col_tuple, [1, 0])

        with self.test_session():
            self.assertAllEqual([[[8, 15, 12],
                                  [21, 36, 27],
                                  [20, 33, 24]],
                                 [[-3, -13, -11],
                                  [10, -5, -9],
                                  [20, 10, 1]]], max_patch.eval())



            self.assertAllEqual([[1, 1],
                                 [2, 0]], idx.eval())


if __name__ == '__main__':
    tf.test.main()
