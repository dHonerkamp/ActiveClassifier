import os.path
import pickle

from tools.utility import Utility
from env.get_data import get_data, random_uk_selection
from phase_config import get_phases
from training import run_phase


def main():
    FLAGS, config = Utility.init()

    if FLAGS.uk_folds:
        n_classes_orig = FLAGS.num_alphabets if FLAGS.dataset == 'omniglot' else FLAGS.num_classes
        FLAGS = random_uk_selection(FLAGS, n_classes_orig)

    # load datasets
    train_data, valid_data, test_data = get_data(FLAGS)

    # phases
    phases = get_phases(FLAGS)

    # store FLAGS for later aggregation of metrics
    with open(os.path.join(FLAGS.path, 'log_flags.pkl'), 'wb') as f:
        pickle.dump(vars(FLAGS), f)

    cp_path = FLAGS.path + "/cp.ckpt"
    initial_phase = True

    for phase in phases:
        if phase['num_epochs'] > 0:
            run_phase(FLAGS, phase, initial_phase, config, cp_path, train_data, valid_data, test_data)
            initial_phase = False


if __name__ == '__main__':
    main()
