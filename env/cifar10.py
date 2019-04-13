import os
import tarfile
import numpy as np
from six.moves import urllib


def get_cifar(data_dir):
    '''
    Source: http://rohitapte.com/2017/04/22/image-recognition-on-the-cifar-10-dataset-using-deep-learning/
    :param data_dir:
    :return:
    '''
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
    if os.path.isfile(data_file):
        pass
    else:
        def progress(block_num, block_size, total_size):
            progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
            print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")

        filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
        tarfile.open(filepath, 'r:gz').extractall(data_dir)

    def load_cifar10data(filename):
        with open(filename, mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')
            features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float32)
            features /= 255
            labels = batch['labels']
            return features, labels

    x_train = np.zeros(shape=(0, 32, 32, 3), dtype=np.float32)
    train_labels = []
    for i in range(1, 5 + 1):
        ft, lb = load_cifar10data(data_dir + 'cifar-10-batches-py/data_batch_' + str(i))
        x_train = np.vstack((x_train, ft))
        train_labels.extend(lb)

    # y_train = tf.one_hot(train_labels, depth=10)
    y_train = np.array(train_labels, dtype=np.int32)

    x_test, test_labels = load_cifar10data(data_dir + 'cifar-10-batches-py/test_batch')
    # y_test = tf.one_hot(test_labels, depth=10)
    y_test = np.array(test_labels, dtype=np.int32)

    return (x_train, y_train), (x_test, y_test)