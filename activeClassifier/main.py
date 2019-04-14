import logging
import time
import tensorflow as tf

from env.env import ImageForagingEnvironment
from tools.utility import Utility
from env.get_data import get_data, random_uk_selection
from models.predRSSM import predRSSM
from visualisation.visualise_predRSSM import Visualization_predRSSM
# from models.activeClassifier import ActiveClassifier
# from visualisation.visualise_ActCl import Visualization_ActCl
from phase_config import get_phases


def evaluate(FLAGS, sess, model, feed, num_batches, writer, visual=None):
    fetch = {'summary'            : model.summary,
             'step'               : model.global_step,
             'metrics'            : model.metrics_update,
             'phase'              : model.phase,
             'y_MC'               : model.y_MC,
             'clf'                : model.classification,
             'fb'                 : model.fb,
             'state_believes'     : model.state_believes,
             }
    batch_values = {k: None for k in ['y_MC', 'clf', 'fb', 'state_believes']}

    for i in range(num_batches):
        out = sess.run(fetch, feed_dict=feed)
        batch_values = Utility.update_batch_stats(batch_values, out, batch_sz=out['y_MC'].shape[0])  # batch_sz of last batch might be smaller

    sess.run(tf.local_variables_initializer())# set streaming_metrics back to zero
    summs = [tf.Summary.Value(tag="batch/" + var, simple_value=avg) for var, avg in zip(model.metrics_names, out["metrics"])]
    batch_summaries = tf.Summary(value=summs)
    writer.add_summary(batch_summaries, out['step'])
    writer.add_summary(out['summary'], out['step'])

    if visual is not None:
        batch_values['y'] = batch_values['y_MC']
        sfx = '_' + writer.get_logdir().split('/')[-1]
        visual.plot_fb(batch_values, suffix=sfx)
        visual.plot_stateBelieves(batch_values, suffix=sfx)

    prefix = writer.get_logdir().split('/')[-1].upper() + ':'
    strs = [item for pair in zip(model.metrics_names, out["metrics"]) for item in pair]
    s = 'step {} - phase {} - {}'.format(out['step'], out['phase'], prefix) + len(strs) // 2 * ' {}: {:.3f}'
    logging.info(s.format(*strs))


def training_loop(FLAGS, sess, model, handles, writers, phase):
    def run_eval():
        # Visual.visualise(sess, eval_feed_train, suffix='_train')
        Visual.visualise(sess, feeds['eval_valid'], suffix='_valid')
        evaluate(FLAGS, sess, model, feeds['eval_train'], FLAGS.batches_per_eval_valid, writers['train'])
        evaluate(FLAGS, sess, model, feeds['eval_valid'], FLAGS.batches_per_eval_valid, writers['valid'], Visual)
        if FLAGS.uk_label:  # just to faster sense if it generalises to UU or not
            evaluate(FLAGS, sess, model, feeds['eval_valid'], FLAGS.batches_per_eval_test, writers['test'], Visual)

    feeds = model.get_feeds(FLAGS, handles)
    Visual = Visualization_predRSSM(model, FLAGS)

    if sess.run(model.global_step) == 0:
        run_eval()

    for epochs_completed in range(phase['num_epochs']):
        # training
        t = time.time()
        train_op = model.get_train_op(FLAGS)

        for i in range(FLAGS.train_batches_per_epoch):
            if i and (i % 100 == 0):
                step, train_summ = sess.run([model.global_step, model.summary], feed_dict=feeds['eval_train'])
                step, valid_summ, loss, acc, T, acc_uk = sess.run([model.global_step, model.summary, model.loss, model.acc, model.avg_T, model.acc_uk], feed_dict=feeds['eval_valid'])
                writers['train'].add_summary(train_summ, global_step=step)
                writers['valid'].add_summary(valid_summ, global_step=step)
                print('{}/{}, loss: {:.3f}, acc: {:.3f}, T: {:.3f}, acc_uk: {:.3f}'.format(i, FLAGS.train_batches_per_epoch, loss, acc, T, acc_uk))
            else:
                sess.run(train_op, feed_dict=feeds['train'])

        logging.info("{} epoch {}: {:.2f}min, {:.3f}s per batch, train op: {}".format(phase['name'], epochs_completed,
                                                                                      (time.time() - t) / 60,
                                                                                      (time.time() - t) / FLAGS.train_batches_per_epoch,
                                                                                      train_op.name))
        # evaluation
        if (epochs_completed % FLAGS.eval_step_interval) == 0:
            run_eval()

    if phase['final_eval']:
        logging.info('FINISHED TRAINING, {} EPOCHS COMPLETED\n'.format(epochs_completed))
        evaluate(FLAGS, sess, model, feeds['eval_test'],  FLAGS.batches_per_eval_test,  writers['test'], Visual)
        Visual.visualise(sess, feeds['eval_test'], suffix='_test')


def main():
    FLAGS, config = Utility.init()

    if FLAGS.uk_folds:
        n_classes_orig = FLAGS.num_alphabets if FLAGS.dataset == 'omniglot' else FLAGS.num_classes
        FLAGS = random_uk_selection(FLAGS, n_classes_orig)

    # load datasets
    train_data, valid_data, test_data = get_data(FLAGS)

    # phases
    phases = get_phases(FLAGS)

    writers = Utility.init_writers(FLAGS)
    cp_path = FLAGS.path + "/cp.ckpt"
    initial_phase = True

    for phase in phases:
        if phase['num_epochs'] == 0:
            continue
        logging.info('Starting phase {}.'.format(phase['name']))
        tf.reset_default_graph()

        if FLAGS.uk_label and (phase['incl_uk'] is False):
            x, y = train_data
            is_uk = (y == FLAGS.uk_label)
            train_data = (x[~is_uk], y[~is_uk])

        with tf.Session(config=config) as sess:
            # Initialise env and model
            env = ImageForagingEnvironment(FLAGS)
            handles = env.intialise(train_data, valid_data, test_data, sess)
            model = predRSSM(FLAGS, env, phase)

            # Weights: initialise / restore from previous phase
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            if initial_phase:
                writers['train'].add_graph(tf.get_default_graph())
                initial_phase = False
            else:
                old_vars = [v[0] for v in tf.train.list_variables(cp_path)]
                variables_can_be_restored = [v for v in tf.global_variables() if v.name.split(':')[0] in old_vars]
                # print(variables_can_be_restored)
                tf.train.Saver(variables_can_be_restored).restore(sess, cp_path)

            # Train
            training_loop(FLAGS, sess, model, handles, writers, phase)
            model.saver.save(sess, cp_path)

    return


if __name__ == '__main__':
    main()
