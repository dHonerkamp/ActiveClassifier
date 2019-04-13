import os
import logging
import numpy as np
from scipy import misc


def get_omniglot_alphabet_labels(alphabets, FLAGS):
    return [v for k, v in FLAGS.labels_dict.items() if k.split('+')[0] in alphabets]


def get_OMNIGLOT(FLAGS, data_path):
    with open(os.path.join(data_path, 'alphabets_train.txt'), 'r') as f:
        alphabets_train_orig = np.array([s for s in f.read().splitlines() if s != ''])
    with open(os.path.join(data_path, 'alphabets_eval.txt'), 'r') as f:
        alphabets_eval_orig = np.array([s for s in f.read().splitlines() if s != '']).astype(alphabets_train_orig.dtype)

    data_path = os.path.join(data_path, "images_all")
    alphabets = np.array(sorted(os.listdir(data_path)))

    if FLAGS.use_orig_alphabet_split:
        alphabets_test = alphabets_eval_orig
        uks = alphabets_test.copy()
    else:
        alphabets_test = alphabets[FLAGS.uk_test_labels]
        uks = FLAGS.uk_test_labels.copy()
    if FLAGS.uk_test_labels_used:
        alphabets_test_used = alphabets_test[:FLAGS.uk_test_labels_used]
    else:
        alphabets_test_used = np.array([])

    if FLAGS.uk_train_labels != -1:
        uks = list(uks) + list(alphabets_train_orig[FLAGS.uk_train_labels])
        if FLAGS.use_orig_alphabet_split:
            alphabets_train_uk = alphabets_train_orig[FLAGS.uk_train_labels]
        else:
            alphabets_train_uk = alphabets[FLAGS.uk_train_labels]

    if FLAGS.use_orig_alphabet_split:
        FLAGS.alphabets_train = np.array(np.setdiff1d(alphabets, uks))
    else:
        train_idc = np.setdiff1d(np.arange(FLAGS.num_alphabets), uks)
        FLAGS.alphabets_train = np.array(alphabets[train_idc])

    labels = [alpha + "+" + character if alpha in FLAGS.alphabets_train
              else "0"
              for alpha in alphabets
              for character in os.listdir(os.path.join(data_path, alpha))]
    labels = sorted(list(set(labels)))

    # "0" comes first in order, so unknowns are assigned numeric label 0, as wanted
    FLAGS.labels_dict = dict(zip(labels, range(len(labels))))
    FLAGS.num_classes = len(labels)

    training = np.array([(os.path.join(data_path, alpha, character, name), FLAGS.labels_dict[alpha + "+" + character])
                         for alpha in FLAGS.alphabets_train
                         for character in os.listdir(os.path.join(data_path, alpha))
                         for name in os.listdir(os.path.join(data_path, alpha, character))])
    test = [(os.path.join(data_path, alpha, character, name), FLAGS.labels_dict["0"])
            for alpha in alphabets_test_used
            for character in os.listdir(os.path.join(data_path, alpha))
            for name in os.listdir(os.path.join(data_path, alpha, character))]

    # training: list of paths with _01.png to _20.png ending for each label
    # put some examples of each label in validation and test
    # chose randomly as the digit refers to the person that drew it -> mix the person over different folds
    a, b = np.random.choice(np.arange(1, 21), 2, replace=False)
    valid_obs = [a]
    test_obs = [b]
    train, valid = [], []
    for obs in training:
        if int(obs[0].split('_')[-1].replace('.png', '')) in valid_obs:
            valid.append(obs)
        elif int(obs[0].split('_')[-1].replace('.png', '')) in test_obs:
            test.append(obs)
        else:
            train.append(obs)

    if FLAGS.uk_train_labels != -1:
        training_uks = np.array([(os.path.join(data_path, alpha, character, name), FLAGS.labels_dict["0"])
                                 for alpha in alphabets_train_uk
                                 for character in os.listdir(os.path.join(data_path, alpha))
                                 for name in os.listdir(os.path.join(data_path, alpha, character))])
        np.random.shuffle(training_uks)
        # add some of the uks to validation, in same proportion as it has knowns
        n_valid = int(len(valid_obs) / 20 * len(training_uks))
        train = np.concatenate([train, training_uks[:-n_valid]])
        valid = np.concatenate([valid, training_uks[-n_valid:]])

    def to_data_tuple(l):
        """Change list of  tuples into tuple of 2 lists"""
        (a, b) = map(list, zip(*l))
        return (np.array(a), np.array(b, dtype=np.int32))

    train = to_data_tuple(train)
    valid = to_data_tuple(valid)
    test = to_data_tuple(test)

    # both [0] now
    FLAGS.uk_test_labels = list(set(test[1]))
    # TODO: uk_test_labels_used MIGHT NEED TO BE A NUMBER, NOT A LIST OF LABELS
    FLAGS.uk_test_labels_used = get_omniglot_alphabet_labels(alphabets_test_used, FLAGS)
    logging.info('Omniglot, total labels: {}, train: {}, test: {}\n'
                 'Labels per set (uk = 0):\n'
                 'train: {}\n'
                 'valid: {}\n'
                 'test:  {}'.format(FLAGS.num_classes, len(set(train[1])), len(set(test[1])), set(train[1]),
                                    set(valid[1]),
                                    set(test[1])))

    return train, valid, test


def get_omni_small(FLAGS, data_path):
    resize = 28
    if os.path.exists(data_path + '/omni{}.npy'.format(resize)):
        omni_img = np.load(data_path + '/omni{}.npy'.format(resize))
    else:
        omni_path = os.path.join(FLAGS.data_dir + "omniglot/", "images_all")
        alphabets = np.array(sorted(os.listdir(omni_path)))

        omni_paths = np.array([os.path.join(omni_path, alpha, character, name)
                               for alpha in alphabets
                               for character in os.listdir(os.path.join(omni_path, alpha))
                               for name in os.listdir(os.path.join(omni_path, alpha, character))])
        omni_img = np.array([misc.imresize(misc.imread(p, flatten=True), size=(resize, resize),
                                           interp='nearest') for p in omni_paths], dtype=np.float32)
        omni_img = omni_img / 255.
        omni_img = np.expand_dims(omni_img, -1)
        omni_img = np.abs(omni_img - 1.)  # change the digit to be 1 and the background 0
        omni_img = np.random.permutation(omni_img)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        np.save(data_path + '/omni{}'.format(resize), omni_img)

    return omni_img
