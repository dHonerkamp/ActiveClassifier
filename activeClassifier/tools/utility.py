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
from collections import deque
from multiprocessing import Process


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


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

        if FLAGS.planner == 'RL':
            assert FLAGS.rl_reward == 'clf'

        if not FLAGS.use_pixel_obs_FE:
            assert (FLAGS.pixel_obs_discrete == 0)

        if FLAGS.uniform_loc10:
            assert FLAGS.planner == 'ActInf'
            assert FLAGS.rl_reward in ['clf', 'G']
            assert ~FLAGS.rnd_first_glimpse  # not completely necessary, but seems to make sense

        # not including any potential unknown class (adjusted in get_data)
        FLAGS.num_classes_kn = FLAGS.num_classes
        # Label representing unknowns (adjusted in get_data)
        FLAGS.uk_label = None

        FLAGS.max_gradient_norm = 5
        FLAGS.loc_dim = 2

        # only for convLSTM
        FLAGS.n_filters_encoder = 8

        if FLAGS.uk_test_labels or FLAGS.uk_folds:
            FLAGS.cache = 0
        assert (FLAGS.uk_test_labels is None) or (type(FLAGS.uk_test_labels) == list)
        assert (FLAGS.uk_train_labels is None) or (type(FLAGS.uk_train_labels) == list)
        assert  (FLAGS.num_uk_test == 0 ) or (FLAGS.uk_test_labels is None), 'Cannot define both num_uk_test and uk_test_labels'
        assert  (FLAGS.num_uk_train == 0 ) or (FLAGS.uk_train_labels is None), 'Cannot define both num_uk_train and uk_train_labels'
        if FLAGS.num_uk_test:
            assert FLAGS.uk_folds
        if FLAGS.uk_cycling:
            assert (FLAGS.uk_folds and (FLAGS.num_uk_test is not None)) or (FLAGS.uk_test_labels is not None), 'Need to include uks in the test set.'

    @staticmethod
    def set_exp_name(FLAGS):
        experiment_name  = '{}gl_{}_{}_bs{}_MC{}_'.format(FLAGS.num_glimpses, FLAGS.planner, FLAGS.beliefUpdate, FLAGS.batch_size, FLAGS.MC_samples)
        experiment_name += 'lr{}dc{}_'.format(FLAGS.learning_rate, FLAGS.learning_rate_decay_factor)
        experiment_name += '{}sc{}_glstd{}_'.format(len(FLAGS.scale_sizes), FLAGS.scale_sizes[0], FLAGS.gl_std)
        experiment_name += '1stGlRnd_' if FLAGS.rnd_first_glimpse else ''
        experiment_name += 'lstd{}to{}Rng{}_'.format(FLAGS.loc_std, FLAGS.loc_std_min, FLAGS.init_loc_rng) if not FLAGS.uniform_loc10 else 'uniformLoc10'
        experiment_name += 'preTr{}{}uk{}_'.format(FLAGS.pre_train_epochs, FLAGS.pre_train_policy, FLAGS.pre_train_uk) if FLAGS.pre_train_epochs else ''
        experiment_name += 'z{sz}{d}{kl}C{c}w{w}_fbN{n}'.format(sz=FLAGS.size_z, d=FLAGS.z_dist, kl=FLAGS.z_B_kl, c=FLAGS.z_B_center, w=FLAGS.z_kl_weight, n=FLAGS.normalise_fb)
        if FLAGS.use_conv:
            experiment_name += '_CNN'
        if FLAGS.convLSTM:
            experiment_name += '_convLSTM'
        if FLAGS.planner == 'ActInf':
            experiment_name += '_c{}a{}p{}{}'.format(FLAGS.prior_preference_c, FLAGS.precision_alpha, FLAGS.prior_preference_glimpses, FLAGS.rl_reward)
            experiment_name += 'pixel{}bins'.format(FLAGS.pixel_obs_discrete) if FLAGS.use_pixel_obs_FE else ''
        if FLAGS.uk_folds:
            experiment_name += '_ukTr{}Te{}U{}Cy{}'.format(FLAGS.num_uk_train, FLAGS.num_uk_test, FLAGS.num_uk_test_used, FLAGS.uk_cycling)
        if FLAGS.binarize_MNIST:
            experiment_name += '_binar'

        if platform != 'win32':
            if FLAGS.freeze_enc:
                experiment_name += '_frzEnc'
            if FLAGS.freeze_policyNet:
                experiment_name += '_frzPol'

        name = os.path.join(FLAGS.exp_folder, experiment_name)

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
        parser.add_argument('--debug', type=int, default=0, choices=[0, 1], help="Debug mode.")
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
        parser.add_argument('-npre', '--pre_train_epochs', type=int, default=0, help='Number of epochs to train generative model only on random locations.')
        parser.add_argument('--pre_train_policy', type=str, default='same', choices=['same', 'random', 'ActInf', 'RL'], help="Pretrain policy. 'Same' to use same as main phase.")
        parser.add_argument('--pre_train_uk', type=int, default=1, choices=[0, 1], help='Whether to include uks during pretrain phase.')
        parser.add_argument('--freeze_enc', type=int, default=None, help='Number of epochs after which to freeze the encoder weights. Set to None to ignore.')
        parser.add_argument('--freeze_policyNet', type=int, default=None, help='Number of epochs after which to freeze the policyNet weights. Set to None to ignore.')
        # locations
        parser.add_argument('--rnd_first_glimpse', type=int, default=1, choices=[0, 1], help='Whether to start with a random glimpse or plan it.')
        parser.add_argument('--uniform_loc10', type=int, default=0, choices=[0, 1], help='Dont learn locations, instead always select from 10 evenly distributed ones. Only for ActInf planner.')
        parser.add_argument('--max_loc_rng', type=float, default=1., help='In what range are the locations allowed to fall? (Max. is -1 to 1)')
        parser.add_argument('--loc_std', type=float, default=0.09, help='Std used to sample locations. Relative to whole image being in range (-1, 1).')
        parser.add_argument('--loc_std_min', type=float, default=0.09, help='Minimum loc_std, decaying exponentially (hardcoded decay rate).')
        parser.add_argument('--init_loc_rng', type=float, default=1., help='Range from which the initial, random location will be sampled. Value between [0, 1].')
        # parser.add_argument('--loc_encoding', type=str, default='cartFiLM', choices=["cartOrig", "cartFiLM", "cartRel", "polarRel"],
        #     help='cartOrig: (x, y), glNet: additive'
        #          'cartFiLM: (x, y), glNet: learning gamma, beta'
        #          'cartRel: diff(x,y) '
        #          'polarRel: r, theta')
        # more important settings
        parser.add_argument('--planner', type=str, default='ActInf', choices=['ActInf', 'RL'], help='Planning strategy.')
        parser.add_argument('--rl_reward', type=str, default='clf', choices=['clf', 'G1', 'G'], help='Rewards for ActInf location policy. For other planners always clf.')
        parser.add_argument('--beliefUpdate', type=str, default='fb', choices=['fb', 'fc', 'RAM'], help='Belief update strategy.')
        parser.add_argument('--normalise_fb', type=int, default=0, choices=[0, 1, 2], help='Use min_normalisation for prediction fb or not. 1: divide by baseline, 1: subtract baseline')
        parser.add_argument('--prior_preference_c', type=int, default=2, help='Strength of preferences for correct / wrong classification.')
        parser.add_argument('--prior_preference_glimpses', type=int, default=-4, help='Penalty for taking more than 4 glimpses (Visual foraging: -2*c).')
        parser.add_argument('--precision_alpha', type=int, default=1, help='Precision constant. Visual foraging_demo: 512')
        parser.add_argument('--use_pixel_obs_FE', type=int, default=0, choices=[0, 1], help='Whether to calculate the expected Free Energy on the pixel outputs.')
        parser.add_argument('--pixel_obs_discrete', type=int, default=0, help='Discretize obs into one-hot. Value=bins (plausibly up to 255), zero to ignore.')
        parser.add_argument('--size_z', type=int, default=32, help='Dimensionality of the hidden states z.')
        parser.add_argument('--z_dist', type=str, default='B', choices=['N', 'B'], help='Distributions of the hidden state. N: normal, B: bernoulli.')
        parser.add_argument('--z_B_kl', type=int, default=22, choices=[20, 21, 22, 212], help='Bernoulli latent code only: type of KL divergence. Corresponds to equations in https://arxiv.org/abs/1611.00712.')
        parser.add_argument('--z_kl_weight', type=float, default=1., help='Weighting the kl term up or down.')
        parser.add_argument('--z_B_center', type=int, default=0, choices=[0, 1], help='Bernoulli latent code only: center the logits.')
        parser.add_argument('--num_hidden_fc', type=int, default=512, help='Standard size of fully connected layers.')
        parser.add_argument('--use_conv', type=int, default=0, choices=[0, 1], help='Whether to use a convolutional encoder/decoder instead of fc.')
        parser.add_argument('--size_rnn', type=int, default=256, help='Size of the RNN cell.')
        parser.add_argument('--convLSTM', type=int, default=0, choices=[0, 1], help='Whether to use a convLSTM + convolution encoder. Leads to ignore size_rnn')
        parser.add_argument('--gl_std', type=int, default=1, help='-1 to learn the glimpse standard deviation in the decoder, value to set it constant.')

        parser.add_argument('-gl', '--num_glimpses', type=int, default=5, help='Number of glimpses the network is allowed to take. If learn_num_glimpses this is the max. number to take.')
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
        # Directly setting UK labels
        parser.add_argument('--uk_test_labels', nargs='+', type=int, default=None,  # [6, 7, 8, 9],
                            help='List of labels to exclude from training set. None to ignore (0 means class 0!). '
                                 '(REMASKING FOR OMNIGLOT MIGHT NOT WORK ANYMORE, AS IT USED TO BE A NUMBER OF ALPHABETS IN THAT CASE).')
        parser.add_argument('--uk_train_labels', nargs='+', type=int, default=None, help='List of labels to set to unknown, but keep in training set. None to ignore.')
        # use uks from different datasets (MNIST_OMNI_notMNIST dataset)
        parser.add_argument('--uk_pct', type=float, default=0.3, help='MNIST_OMNI_notMNIST only: share of the dataset to be added as uks (resulting total observations = 100*(1 + uk_pct)%')
        # uk cycling
        # TODo: ATM DON'T HAVE ANY UK IN THE VALIDATION SET. BUT HAVE TO KEEP THE IS_TRAINING CONDITION, O/W WILL DO IT DURING TEST AS WELL
        parser.add_argument('--uk_cycling', type=int, default=0, help='Whether to mask random known classes as uk each batch (whose predictions will be masked). Value determines how many uk classes to draw each batch.')
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
            assert out[s].shape[:2].count(batch_sz) == 1, 'Ambigous shape, multiple dim with size batch_sz: {}. ' \
                                                      'Make sure num_glimpses != batch_sz.'.format(out[s].shape)
            dim0 = out[s].shape[0]
            batch_major = (dim0 == batch_sz)

            if stats[s] is None:  # initial
                stats[s] = out[s]
            else:
                ax = 0 if batch_major else 1
                stats[s] = np.append(stats[s], out[s], axis=ax)
        return stats

    @staticmethod
    def configure_logging(name, path, debug):
        file_handler = logging.FileHandler(os.path.join(path, 'log.log'))

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(ElapsedFormatter())

        level = logging.DEBUG if debug else logging.INFO

        logging.basicConfig(level=level,
                            format='%(asctime)s %(name)s %(message)s',
                            handlers=[stream_handler,
                                      file_handler])

        return logging.getLogger(name)

    @staticmethod
    def init(experiment_name=None):
        # Parsing experimental set up
        FLAGS, unparsed = Utility.parse_arg()
        Utility.auto_adjust_flags(FLAGS)
        if experiment_name is None:
            experiment_name = Utility.set_exp_name(FLAGS)
        FLAGS.experiment_name = experiment_name

        folder = '{d}{tsz}{resz}{uk}'.format(d=FLAGS.dataset, tsz=FLAGS.translated_size or '', resz=FLAGS.img_resize or '',
                                             uk = '_UK' if FLAGS.num_uk_test or FLAGS.uk_test_labels else '')
        FLAGS.path = os.path.join(FLAGS.summaries_dir, folder, FLAGS.experiment_name)
        os.makedirs(FLAGS.path, exist_ok=True)

        logger = Utility.configure_logging(__name__, FLAGS.path, FLAGS.debug)
        logger.info('CURRENT MODEL: ' + FLAGS.experiment_name + '\n')

        # # log tf to a file as well
        # tf_logging = logging.getLogger('tensorflow')
        # tf_logging.setLevel(logging.DEBUG)
        #
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # fh = logging.FileHandler(os.path.join(FLAGS.path, 'tf.log'))
        # fh.setLevel(logging.DEBUG)
        # fh.setFormatter(formatter)
        # tf_logging.addHandler(fh)

        # os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'

        if unparsed and not 'activeClassifier.test' in unparsed[0]:
            raise ValueError('UNPARSED: {}'.format(unparsed))
        logger.info(sys.argv)

        # reproducibility (not guaranteed if run on GPU or different platforms)
        Utility.set_seeds()

        # session config
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
        return "{} {}: {}".format(elapsed, record.name, record.getMessage())


class Proc_Queue(deque):
    """
    Custom queue to manage the number of (plotting) processes run in parallel to training.
    When passing max_len first wait for a process to finish, then start the next."""
    def __init__(self, max_len, max_wait=60):
        super().__init__()
        self.max_len = max_len
        self.max_wait = max_wait

    def add_proc(self, target, args, name=None):
        """If queue already full: wait for a process to terminate, then start x and append it"""
        if self.max_len == 0:  # don't add any processes, execute target and return
            target(*args)
            return

        if len(self) >= self.max_len:
            to_remove = None
            t = time.time()
            while not to_remove:
                for elem in self:
                    if not elem.is_alive():
                        elem.join()
                        to_remove = elem
                        break
                else:
                    time.sleep(1)
                    if self.max_wait and ((time.time() - t) > 60):
                        raise ValueError('Subprocesses took more than {} to finish: {}'.format(self.max_wait, self))
            self.remove(to_remove)

        logging.debug('Process qeue length:', len(self), self)
        proc = Process(target=target, args=args, name=name)
        self.append(proc)
        proc.start()

    def cleanup(self):
        while len(self):
            self.popleft().join()
