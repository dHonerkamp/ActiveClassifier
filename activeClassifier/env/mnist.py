import numpy as np
import tensorflow as tf

def get_MNIST(FLAGS):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    n_valid = 5000
    x_valid, y_valid = x_train[:n_valid], y_train[:n_valid]
    x_train, y_train = x_train[n_valid:], y_train[n_valid:]

    # shuffle so MNIST_OMNI_notMNIST can just split them at will
    def shuffle(x, y):
        idx = np.random.permutation(x.shape[0])
        return x[idx], y[idx]

    x_train, y_train = shuffle(x_train, y_train)
    x_valid, y_valid = shuffle(x_valid, y_valid)
    x_test, y_test   = shuffle(x_test, y_test)

    train = (np.reshape(x_train, [x_train.shape[0]] + FLAGS.img_shape), np.array(y_train, dtype=np.int32))
    valid = (np.reshape(x_valid, [x_valid.shape[0]] + FLAGS.img_shape), np.array(y_valid, dtype=np.int32))
    test  = (np.reshape(x_test, [x_test.shape[0]]   + FLAGS.img_shape), np.array(y_test, dtype=np.int32))

    if FLAGS.binarize_MNIST:
        def binarize(images, threshold=0.1):
            return (threshold < images).astype('float32')

        train = (binarize(train[0]), train[1])
        valid = (binarize(valid[0]), valid[1])
        test  = (binarize(test[0]), test[1])

    return train, valid, test