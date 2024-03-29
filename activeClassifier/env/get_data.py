import logging
import re
import numpy as np

from activeClassifier.env.mnist import get_MNIST
from activeClassifier.env.cifar10 import get_cifar
from activeClassifier.env.notMNIST import notMNIST
from activeClassifier.env.omniglot import get_omni_small, get_OMNIGLOT

logger = logging.getLogger(__name__)

### Unknown FLAGS
## When masking parts of a single dataset:
# FLAGS.num_uk_test: int, generates FLAGS.uk_test_labels
# FLAGS.uk_test_labels: list of uk labels
# FLAGS.uk_test_labels_used: int, how many unknowns to keep
# FLAGS.num_uk_train: int, generates FLAGS.uk_train_labels
# FLAGS.uk_train_labels: list of uk labels
## Using uks from different datasets (MNIST_OMNI_notMNIST dataset):
# FLAGS.uk_pct: share of the dataset to be added as uks (resulting total observations = 100*(1 + uk_pct)%

# def create_class_mapping_uk0(num_classes, uks):
#     uks = np.array(uks)
#     mapping = {}
#     uk_label = 0
#
#     for c in range(num_classes):
#         if c in uks:
#             mapping[c] = 0
#         else:
#             smaller_uks = (uks < c).sum()
#             mapping[c] = c - smaller_uks + 1   # +1 as 0 will be "unknown"
#
#     return mapping, uk_label

def random_uk_selection(FLAGS, num_classes):
    # Scheirer - MNIST: 6 knowns, varying openness with other 4, 20-folds each
    # -> num_uk_train = 6, num_uk_test_used = [1, 4], uk_runs = 20
    uks = np.random.choice(num_classes, size=num_classes, replace=False)
    FLAGS.uk_train_labels = list(uks[:FLAGS.num_uk_train])
    FLAGS.uk_test_labels = list(uks[FLAGS.num_uk_train:FLAGS.num_uk_train + FLAGS.num_uk_test])

    logger.info('\tuk_train_labels: {}\n'
                 '\tuk_test_labels: {}\n'
                 '\tuk_test_labels used: {}'.format(FLAGS.uk_train_labels, FLAGS.uk_test_labels,
                                                    FLAGS.num_uk_test_used))

    assert (FLAGS.uk_test_labels is None) or (type(FLAGS.uk_test_labels) == list)
    assert (FLAGS.uk_train_labels is None) or (type(FLAGS.uk_train_labels) == list)

    return FLAGS


def create_class_mapping_ukMax(num_classes, uks):
    if uks:
        uks = np.array(uks)
        mapping = {}
        reverse_mapping = {}  # to display the correct original label in visualisation, uk = -1

        new_num_classes = num_classes - (len(uks) - 1)
        new_num_classes_kn = new_num_classes - 1
        uk_label = new_num_classes - 1 # -1 because of zero-indexing

        for c in range(num_classes):
            if c in uks:
                mapping[c] = uk_label
                reverse_mapping[uk_label] = 'uk'
            else:
                smaller_uks = (uks < c).sum()
                mapping[c] = c - smaller_uks
                reverse_mapping[c - smaller_uks] = c
    else:
        new_num_classes, new_num_classes_kn = num_classes, num_classes
        mapping, uk_label = None, None
        reverse_mapping = {c: c for c in range(num_classes)}

    reverse_mapping[-1] = 'No dec'  # clf outputs -1 if no classification decision is made
    return mapping, reverse_mapping, uk_label, new_num_classes, new_num_classes_kn


def remove_labels(x, y, labels):
    ix = np.isin(y, labels)
    return x[~ix], y[~ix]


def move_unknowns_into_test_set(FLAGS, train, valid, test):
    """
    move all obs that belong to unknown_test-classes into test,
    setting according labels to highest new class value"""
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y   = test

    # remove unused test uks from valid, test
    uk_unused = FLAGS.uk_test_labels[FLAGS.num_uk_test_used:]
    test_x, test_y   = remove_labels(test_x, test_y, uk_unused)
    valid_x, valid_y = remove_labels(valid_x, valid_y, uk_unused)

    # remove test uks from train
    train_x, train_y = remove_labels(train_x, train_y, FLAGS.uk_test_labels)
    uks = FLAGS.uk_test_labels.copy()

    if FLAGS.uk_train_labels:
        uks += FLAGS.uk_train_labels.copy()

        # remove train uks from valid, test
        test_x, test_y   = remove_labels(test_x, test_y, FLAGS.uk_train_labels)
        valid_x, valid_y = remove_labels(valid_x, valid_y, FLAGS.uk_train_labels)

    class_mapping, label_remapping, uk_label, FLAGS.num_classes, FLAGS.num_classes_kn = create_class_mapping_ukMax(FLAGS.num_classes, uks)
    train_y = np.vectorize(class_mapping.get)(train_y)
    valid_y = np.vectorize(class_mapping.get)(valid_y)
    test_y  = np.vectorize(class_mapping.get)(test_y)

    return  (train_x, train_y), (valid_x, valid_y), (test_x, test_y), label_remapping, uk_label


# def relabel_notOmniglot_knowns(labels, uk_test_labels, uk_train_labels):
#     labels = copy.deepcopy(labels)
#     uk_idx = labels[labels == 0]
#     labels -= 1
#
#     uks_orig = uk_test_labels
#     if uk_train_labels:
#         uks_orig += uk_train_labels
#
#     for uk in sorted(uks_orig):
#         labels[uk <= labels] += 1
#
#     # put unknowns onto 99, so 0 can be 0 again
#     labels[uk_idx] = 99
#     labels[labels == -1] = 0
#
#     return labels


# def mask_train_unknown(train, valid, unknown_label):
#     # just to be sure I don't change original labels (need to reset label at each change)
#     train_labels = copy.deepcopy(train[1])
#     valid_obs    = copy.deepcopy(valid[0])
#     valid_labels = copy.deepcopy(valid[1])
#
#     train_ix = np.isin(train[1], unknown_label)
#     valid_ix = np.isin(valid[1], unknown_label)
#
#     # mask in train data
#     train_labels[train_ix] = 0
#     # remove from validation data
#     valid_labels = valid_labels[~valid_ix]
#     valid_obs    = valid_obs[~valid_ix]
#
#     return (train[0], train_labels), (valid_obs, valid_labels)


def get_data(FLAGS):
    '''
    :return: train, valid, test, each a tuple of (images, sparse labels)
    '''
    train, valid, test = None, None, None

    data_path = FLAGS.data_dir + FLAGS.dataset + "/"

    if FLAGS.dataset == 'MNIST':
        train, valid, test = get_MNIST(FLAGS)
    elif FLAGS.dataset == "MNIST_cluttered":
        NUM_FILES = 100000
        filenames = np.array([data_path + "img_{}.png".format(i) for i in range(1, NUM_FILES+1)])
        with open(data_path + "labels.txt",'r') as file:
            labels = file.read()
        labels = re.sub(r"\d+\t", "", labels).split('\n')
        labels = np.array(labels, dtype=np.int32)

        train = (filenames[:80000], labels[:80000])
        valid = (filenames[80000:90000], labels[80000:90000])
        test =  (filenames[90000:], labels[90000:])
    elif FLAGS.dataset == "cifar10":
        (x_train, y_train), test = get_cifar(data_path)# cifar10.load_data()
        train = (x_train[:-10000], y_train[:-10000])
        valid = (x_train[-10000:], y_train[-10000:])
    elif FLAGS.dataset == "omniglot":
        train, valid, test = get_OMNIGLOT(FLAGS, data_path)
        raise ValueError('New uk masking not yet implemented.')
    elif FLAGS.dataset == 'MNIST_OMNI_notMNIST':
        train, valid, test = get_MNIST(FLAGS)
        len_tr = int(FLAGS.uk_pct * train[0].shape[0])
        len_valid = int(FLAGS.uk_pct * valid[0].shape[0])
        len_test = int(FLAGS.uk_pct * test[0].shape[0])

        # uk will be an additional class with the highest label
        FLAGS.num_classes += 1
        _, FLAGS.class_remapping, FLAGS.uk_label, FLAGS.num_classes, FLAGS.num_classes_kn = create_class_mapping_ukMax(FLAGS.num_classes, uks=[FLAGS.num_classes - 1])

        # add resized OMNIGLOT as unknowns to train and validation set
        def get_uk_y(length):
            return np.full([length], FLAGS.uk_label, dtype=np.int32)
        omni_img, omni_lbls = get_omni_small(FLAGS, data_path)
        train = (np.concatenate([train[0], omni_img[:len_tr]], axis=0),
                 np.concatenate([train[1], get_uk_y(len_tr)], axis=0))
        valid = (np.concatenate([valid[0], omni_img[len_tr:len_tr + len_valid]], axis=0),
                 np.concatenate([valid[1], get_uk_y(len_valid)], axis=0))
        FLAGS.num_uk_train = len(np.unique(omni_lbls[len_tr:len_tr + len_valid]))

        # add notMNIST as unknowns to test set
        notmnist = notMNIST(FLAGS.data_dir, test_only=True)
        notMNIST_x, notMNIST_y = notmnist.test
        test = (np.concatenate([test[0], notMNIST_x[:len_test]], axis=0),
                np.concatenate([test[1], get_uk_y(len_test)], axis=0))
        FLAGS.num_uk_test = len(np.unique(notMNIST_y[:len_test]))

        # assure FLAGS.uk_pct is not set so high that we don't have enough unknown observations
        assert (len_tr + len_valid) <= omni_img.shape[0]
        assert len_test <= notMNIST_x.shape[0]

    # mask test unknowns to be label 0 and adjust all other labels
    if (FLAGS.uk_test_labels) and (FLAGS.dataset != 'omniglot'):
        train, valid, test, FLAGS.class_remapping, FLAGS.uk_label = move_unknowns_into_test_set(FLAGS, train, valid, test)
    elif FLAGS.dataset  == 'MNIST_OMNI_notMNIST':
        pass
    else:
        _, FLAGS.class_remapping, FLAGS.uk_label, FLAGS.num_classes, FLAGS.num_classes_kn = create_class_mapping_ukMax(FLAGS.num_classes, uks=[])

    logger.info("Obs per dataset: {}, {}, {}".format(len(train[0]), len(valid[0]), len(test[0])))
    logger.info('(Adapted) labels per set (uk = {}):\n'
                 'train: {}\n'
                 'valid: {}\n'
                 'test:  {}\n'
                 'total classes: {}, kn classes: {}'.format(FLAGS.uk_label, set(train[1]), set(valid[1]), set(test[1]), FLAGS.num_classes, FLAGS.num_classes_kn))

    # ensure flags are set correctly
    assert FLAGS.num_classes - FLAGS.num_classes_kn in [0, 1]
    num_classes_train = len(set(train[1]))
    assert ((FLAGS.num_classes == num_classes_train)  # no uks
            or (FLAGS.num_classes == num_classes_train + 1 and FLAGS.num_uk_train == 0 and not FLAGS.uk_train_labels)  # no train uks
            or (FLAGS.num_classes == num_classes_train + 1 and FLAGS.uk_cycling))
    if FLAGS.uk_label and not FLAGS.uk_cycling and (FLAGS.num_uk_train or FLAGS.uk_train_labels or (FLAGS.dataset == 'MNIST_OMNI_notMNIST')):
        assert FLAGS.num_classes_kn == num_classes_train - 1, 'kn: {}, train: {}'.format(FLAGS.num_classes_kn, num_classes_train)
    else:
        assert FLAGS.num_classes_kn == num_classes_train, 'kn: {}, train: {}'.format(FLAGS.num_classes_kn, num_classes_train)

    FLAGS.train_batches_per_epoch = np.ceil(train[0].shape[0] / FLAGS.batch_size).astype(int)
    FLAGS.batches_per_eval_valid  = np.ceil(valid[0].shape[0] / FLAGS.batch_size).astype(int)
    FLAGS.batches_per_eval_test   = np.ceil(test[0].shape[0]  / FLAGS.batch_size).astype(int)

    FLAGS.train_data_shape = (train[0].shape, train[1].shape)
    FLAGS.valid_data_shape = (valid[0].shape, valid[1].shape)
    FLAGS.test_data_shape  = (test[0].shape, test[1].shape)
    FLAGS.data_dtype       = (train[0].dtype, train[1].dtype)

    return train, valid, test
