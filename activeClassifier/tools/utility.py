import os
import sys
from sys import platform
from datetime import timedelta
import logging
import time
import argparse
import random as rn
import tensorflow as tf
import numpy as np


class Utility(object):
    """
    A class for storing a number of useful functions
    """

    def __init__(self, debug=True):
        pass

    @staticmethod
    def auto_adjust_flags(FLAGS):
        if FLAGS.dataset == "omniglot":
            FLAGS.img_shape        = [105, 105, 1]
            FLAGS.padding          = "zero"
            FLAGS.use_orig_alphabet_split = True
            FLAGS.num_alphabets    = 30 if FLAGS.use_orig_alphabet_split else 50

            assert (type(FLAGS.uk_test_labels) == int) and (type(FLAGS.uk_test_labels_used) == int), 'Provide number of labels for omniglot, not a list.'
        if FLAGS.dataset == "MNIST_cluttered":
            FLAGS.img_shape        = [100, 100, 1]
            FLAGS.padding          = "zero"
            FLAGS.num_classes      = 10
        if FLAGS.dataset in ["MNIST", "MNIST_OMNI_notMNIST"]:
            FLAGS.img_shape        = [28, 28, 1]
            FLAGS.padding          = "zero"
            FLAGS.num_classes      = 10
        if FLAGS.dataset == 'MNIST_OMNI_notMNIST':
            FLAGS.uk_test_labels = []
        if FLAGS.dataset == "cifar10":
            FLAGS.img_shape        = [32, 32, 3]
            FLAGS.padding          = "uniform"
            FLAGS.num_classes      = 10

        # not including any potential unknown class (adjusted in get_data)
        FLAGS.num_classes_kn = FLAGS.num_classes
        # Label representing unknowns (adjusted in get_data)
        FLAGS.uk_label = None

        FLAGS.max_gradient_norm = 5
        FLAGS.loc_dim = 2

        if FLAGS.uk_test_labels or FLAGS.uk_folds:
            FLAGS.cache = 0
        assert (FLAGS.uk_test_labels is None) or (type(FLAGS.uk_test_labels) == list)
        assert (FLAGS.uk_train_labels is None) or (type(FLAGS.uk_train_labels) == list)

    @staticmethod
    def set_exp_name(FLAGS):
        experiment_name = '{}gl_{}_{}_bs{}_MC{}_preTr{}{}uk{}_dcay{}_lr{}_{}sc{}_lstd{}_glstd{}_z{}_fbN{}'.format(
                FLAGS.num_glimpses, FLAGS.planner, FLAGS.beliefUpdate, FLAGS.batch_size, FLAGS.MC_samples,
                FLAGS.pre_train_epochs, FLAGS.pre_train_policy, FLAGS.pre_train_uk,
                FLAGS.learning_rate_decay_factor, FLAGS.learning_rate,
                len(FLAGS.scale_sizes), FLAGS.scale_sizes[0], FLAGS.loc_std, FLAGS.gl_std, FLAGS.size_z, FLAGS.normalise_fb)
        if FLAGS.use_conv:
            experiment_name += '_CNN'
        if FLAGS.uk_folds:
            experiment_name += '_uk{}_{}'.format(FLAGS.num_uk_train, FLAGS.num_uk_test)
        if FLAGS.binarize_MNIST:
            experiment_name += '_binar'

        if platform != 'win32':
            if FLAGS.freeze_enc:
                experiment_name += '_frzEnc'
            if FLAGS.freeze_policyNet:
                experiment_name += '_frzPol'

        name = os.path.join(FLAGS.exp_folder, experiment_name)
        logging.info('CURRENT MODEL: ' + name + '\n')

        return name

    @staticmethod
    def parse_arg():
        """
        Parsing input arguments.

        Returns:
            Parsed data
        """
        parser = argparse.ArgumentParser(description='[*] ActiveClassifier.')

        # paths and boring stuff
        # parser.add_argument('--start_checkpoint', type=str, default='', help='CLOSE TENSORBOARD! If specified, restore this pre-trained model before any training.')
        parser.add_argument('-f', '--exp_folder', type=str, default='', help="Used for log folder.")
        parser.add_argument('--full_summary', type=int, default=0, choices=[0, 1], help='Include all gradients and variables in summary.')
        parser.add_argument('--data_dir', type=str, default='data/', help="Where the data is stored")
        parser.add_argument('--summaries_dir', type=str, default='logs', help='Where to save summary logs for TensorBoard.')
        parser.add_argument('--num_parallel_preprocess', type=int, default=4, help="Parallel processes during pre-processing (on CPU)")
        parser.add_argument('--cache', type=int, default=1, help="Whether to cache the dataset.")
        parser.add_argument('--gpu_fraction', type=float, default=0., help="Limit GPU-RAM.")
        parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default=-1, help="Number of the gpu to use. -1 to ignore.")
        # parser.add_argument('--debug', type=str, default=True, help="Debug mode.")
        # parser.add_argument('--folds', type=int, default=1, help='How often to run the model. uk_folds takes precedence.')
        # parser.add_argument('--f_vis', type=int, default=5, help='Stop visualizing and checkpointing after this many folds.')
        parser.add_argument('--visualisation_level', type=int, default=2, choices=[0, 1, 2], help='Level of visuals / plots to be produced. 2: all plots, 0: no plots.')
        parser.add_argument('--eval_step_interval', type=int, default=1, help='How often to evaluate the training results. In epochs.')
        # learning rates
        parser.add_argument('--learning_rate', type=float, default=0.001, help='How large a learning rate to use when training.')
        parser.add_argument('--learning_rate_decay_factor', type=float, default=0.97, help='1 for no decay.')
        parser.add_argument('--min_learning_rate', type=float, default=0.0001, help='Minimal learning rate.')
        # parser.add_argument('--learning_rate_RL', type=float, default=1, help='Relative weight of the RL objective.')
        # standard parameters
        parser.add_argument('-b', '--batch_size', type=int, default=64, help='How many items to train with at once.')
        parser.add_argument('--MC_samples', type=int, default=10, help='Number of Monte Carlo Samples per image.')
        parser.add_argument('-e', '--num_epochs', type=int, default=8, help='Number of training epochs.')
        parser.add_argument('--pre_train_epochs', type=int, default=0, help='Number of epochs to train generative model only on random locations.')
        parser.add_argument('--pre_train_policy', type=str, default='same', choices=['same', 'random', 'ActInf', 'RL'], help="Pretrain policy. 'Same' to use same as main phase.")
        parser.add_argument('--pre_train_uk', type=int, default=1, choices=[0, 1], help='Whether to include uks during pretrain phase.')
        parser.add_argument('--freeze_enc', type=int, default=None, help='Number of epochs after which to freeze the encoder weights. Set to None to ignore.')
        parser.add_argument('--freeze_policyNet', type=int, default=None, help='Number of epochs after which to freeze the policyNet weights. Set to None to ignore.')
        # locations
        parser.add_argument('--max_loc_rng', type=float, default=1., help='In what range are the locations allowed to fall? (Max. is -1 to 1)')
        parser.add_argument('--loc_std', type=float, default=0.09, help='Std used to sample locations. Relative to whole image being in range (-1, 1).')
        # parser.add_argument('--loc_encoding', type=str, default='cartFiLM', choices=["cartOrig", "cartFiLM", "cartRel", "polarRel"],
        #     help='cartOrig: (x, y), glNet: additive'
        #          'cartFiLM: (x, y), glNet: learning gamma, beta'
        #          'cartRel: diff(x,y) '
        #          'polarRel: r, theta')
        # more important settings
        parser.add_argument('--planner', type=str, default='ActInf', choices=['ActInf', 'RL'], help='Planning strategy.')
        parser.add_argument('--beliefUpdate', type=str, default='fb', choices=['fb', 'fc', 'RAM'], help='Belief update strategy.')
        parser.add_argument('--normalise_fb', type=int, default=0, choices=[0, 1], help='Use min_normalisation for prediction fb or not.')
        parser.add_argument('--prior_preference_c', type=int, default=2, help='Strength of preferences for correct / wrong classification.')
        parser.add_argument('--size_z', type=int, default=32, help='Dimensionality of the hidden states z.')
        parser.add_argument('--num_hidden_fc', type=int, default=512, help='Standard size of fully connected layers.')
        parser.add_argument('--use_conv', type=int, default=0, choices=[0, 1], help='Whether to use a convolutional encoder/decoder instead of fc.')
        parser.add_argument('--size_rnn', type=int, default=256, help='Size of the RNN cell.')
        parser.add_argument('--gl_std', type=int, default=1, help='-1 to learn the glimpse standard deviation in the decoder, value to set it constant.')

        parser.add_argument('--num_glimpses', type=int, default=5, help='Number of glimpses the network is allowed to take. If learn_num_glimpses this is the max. number to take.')
        parser.add_argument('--resize_method', type=str, default="BILINEAR", choices=['AVG', 'BILINEAR', 'BICUBIC'], help='Method used to downsize the larger retina scales. AVG: average pooling')
        parser.add_argument('--scale_sizes', nargs='+', type=int, default=[8],
                            help='List of scale dimensionalities used for retina network (size of the glimpses). Resolution gets reduced to first glimpses size.'
                                 'Smallest to largest scale. Following scales must be a multiple of the first. Might not work for uneven scale sizes!')
        # dataset
        parser.add_argument('-d', '--dataset', type=str, default='MNIST', choices=['MNIST', 'MNIST_cluttered', 'MNIST_OMNI_notMNIST', 'cifar10', 'omniglot'], help='What dataset to use.')
        parser.add_argument('--translated_size', type=int, default=0, help='Size of the canvas to translate images on.')
        parser.add_argument('--img_resize', type=int, default=0, help='Pixels to which to resize the images to.')
        parser.add_argument('--binarize_MNIST', type=int, choices=[0, 1], default=0, help='Binarize MNIST.')

        # UK meta-run settings
        parser.add_argument('--uk_folds', type=int, default=0, help='Number of folds for uk setting. Scheirer et al.: 20')
        parser.add_argument('--num_uk_train', type=int, default=0, help='Number of unknown classes during training. Scheirer et al.: 0')
        parser.add_argument('--num_uk_test', type=int, default=0, help='Number of unknown classes during test. Scheirer et al.: [0-4]')
        parser.add_argument('--num_uk_test_used', type=int, default=0, help='Varying openness. Number of "uk_test_labels" that are actually included in the test set. Starting from the last label in uk_test_labels. Only relevant if uk_test_labels != -1.')
        # parser.add_argument('--uk_pct', type=float, default=.5, help='Share of unknowns to add in. Only on MNIST_OMNI_notMNIST atm.')
        # Directly setting UK labels
        parser.add_argument('--uk_test_labels', nargs='+', type=int, default=None,  # [6, 7, 8, 9],
                            help='List of labels to exclude from training set. None to ignore (0 means class 0!). '
                                 '(REMASKING FOR OMNIGLOT MIGHT NOT WORK ANYMORE, AS IT USED TO BE A NUMBER OF ALPHABETS IN THAT CASE).')
        parser.add_argument('--uk_train_labels', nargs='+', type=int, default=None, help='List of labels to set to unknown, but keep in training set. None to ignore.')
        # use uks from different datasets (MNIST_OMNI_notMNIST dataset)
        parser.add_argument('--uk_pct', type=float, default=0.3, help='Share of the dataset to be added as uks (resulting total observations = 100*(1 + uk_pct)%')
        # parser.add_argument('--uk_cycle_schedule', type=int, default=0, choices=[0, 1], help='Whether to use the "open sed schedule" from the thesis. Set uk_train_labels to 0!')
        # parser.add_argument('--num_uks_per_cycle', type=int, default=2, help='Number of classes to mask each epoch, if cycling.')
        # parser.add_argument('--punish_uk_wrong', type=float, default=0, help='If not 0: Reward of its value for not classifying an unknown as unknown.')

        FLAGS, unparsed = parser.parse_known_args()

        return FLAGS, unparsed

    @staticmethod
    def set_seeds(s=42):
        """
        https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        """
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(s)
        rn.seed(s)
        tf.set_random_seed(s)

    @staticmethod
    def init_writers(FLAGS):
        return {'train': tf.summary.FileWriter(FLAGS.path + '/train'),
                'valid': tf.summary.FileWriter(FLAGS.path + '/valid'),
                'test': tf.summary.FileWriter(FLAGS.path + '/test')}

    @staticmethod
    def update_batch_stats(stats, out, batch_sz):
        """
        Currently only for axis 0 or 1 = batch_sz
        """
        for s in stats.keys():
            dim0 = out[s].shape[0]
            batch_major = (dim0 == batch_sz)

            if stats[s] is None:  # initial
                stats[s] = out[s]
            else:
                ax = 0 if batch_major else 1
                stats[s] = np.append(stats[s], out[s], axis=ax)
        return stats

    @staticmethod
    def init():
        # add custom formatter to root logger for simple demonstration
        handler = logging.StreamHandler()
        handler.setFormatter(ElapsedFormatter())
        logging.getLogger().addHandler(handler)

        logging.getLogger().setLevel(logging.INFO)
        logging.info(sys.argv)
        # os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'

        Utility.set_seeds()

        # Parsing experimental set up
        FLAGS, unparsed = Utility.parse_arg()
        if unparsed:
            logging.info('UNPARSED: {}'.format(unparsed))
        # set img_shape, padding, num_classes according to dataset (ignoring cl inputs!)
        Utility.auto_adjust_flags(FLAGS)
        experiment_name = Utility.set_exp_name(FLAGS)

        t_sz = (str(FLAGS.translated_size) if FLAGS.translated_size else "")
        resz = (str(FLAGS.img_resize if FLAGS.img_resize else ""))
        FLAGS.path = os.path.join(FLAGS.summaries_dir, FLAGS.dataset + t_sz + resz, experiment_name)
        os.makedirs(FLAGS.path, exist_ok=True)

        with open(os.path.join(FLAGS.path, 'argparse.txt'),'w+') as f:
            f.write(' '.join(sys.argv))

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        if FLAGS.gpu_fraction:
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
            # config.gpu_options.allow_growth = True
        if FLAGS.CUDA_VISIBLE_DEVICES != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.CUDA_VISIBLE_DEVICES)

        return FLAGS, config


class ElapsedFormatter:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def format(self, record):
        elapsed_seconds = record.created - self.last_time
        # using timedelta here for convenient default formatting
        elapsed = timedelta(seconds=np.ceil(elapsed_seconds))  # ceil to not display microseconds
        return "{} {}".format(elapsed, record.getMessage())
