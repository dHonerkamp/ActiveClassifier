import os.path
import pickle

from activeClassifier.tools.utility import Utility
from activeClassifier.env.get_data import get_data, random_uk_selection
from activeClassifier.phase_config import get_phases
from activeClassifier.training import run_phase


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

    writers = Utility.init_writers(FLAGS)
    initial_phase = True

    for phase in phases:
        if phase['num_epochs'] > 0:
            run_phase(FLAGS, phase, initial_phase, config, writers, train_data, valid_data, test_data)
            initial_phase = False

    for writer in writers.values():
        writer.close()


if __name__ == '__main__':
    main()
