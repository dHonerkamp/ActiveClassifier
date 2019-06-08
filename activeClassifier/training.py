import sys
import logging
import time
import tensorflow as tf
from sklearn.metrics import f1_score

from env.env import ImageForagingEnvironment
from tools.utility import Utility, Proc_Queue
from models.predRSSM import predRSSM
from visualisation.visualise_predRSSM import Visualization_predRSSM
# from models.activeClassifier import ActiveClassifier
# from visualisation.visualise_ActCl import Visualization_ActCl

# only to display G matrix
import numpy as np
np.set_printoptions(precision=2)

logger = logging.getLogger(__name__)


def batch_plotting(visual, batch_values, sfx):
    visual.plot_fb(batch_values, sfx)
    visual.plot_stateBelieves(batch_values, sfx)


def evaluate(FLAGS, sess, model, feed, num_batches, writer, visual=None, proc_queue=None):
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

    f1 = f1_score(batch_values['y_MC'], batch_values['clf'], labels=None, average="macro")
    summs.append(tf.Summary.Value(tag='batch/Main/f1_macro', simple_value=f1))

    batch_summaries = tf.Summary(value=summs)
    writer.add_summary(batch_summaries, out['step'])
    writer.add_summary(out['summary'], out['step'])

    if visual is not None:
        batch_values['y'] = batch_values['y_MC']
        sfx = '_' + writer.get_logdir().split('/')[-1]
        proc_queue.add_proc(target=batch_plotting, args=(visual, batch_values, sfx), name='batch_plotting{}'.format(sfx))

    prefix = writer.get_logdir().split('/')[-1].upper() + ':'
    strs = [item for pair in sorted(zip(model.metrics_names, out["metrics"])) for item in pair] + ['f1', f1]
    s = 'step {} - phase {} - {}'.format(out['step'], out['phase'], prefix) + len(strs) // 2 * ' {}: {:.3f}'
    logger.info(s.format(*strs))


def run_eval(FLAGS, sess, feeds, model, writers, visual, proc_queue):
    # d = visual.eval_feed(sess, feeds['eval_train'], model)
    # Visual.visualise(d, suffix='_train')
    d = visual.eval_feed(sess, feeds['eval_valid'], model)
    proc_queue.add_proc(target=visual.visualise, args=(d, '_valid'), name='visualise_valid')

    evaluate(FLAGS, sess, model, feeds['eval_train'], FLAGS.batches_per_eval_valid, writers['train'], visual, proc_queue)
    evaluate(FLAGS, sess, model, feeds['eval_valid'], FLAGS.batches_per_eval_valid, writers['valid'], visual, proc_queue)
    if FLAGS.uk_label:  # just to faster sense if it generalises to UU or not
        evaluate(FLAGS, sess, model, feeds['eval_test'], FLAGS.batches_per_eval_test, writers['test'], visual, proc_queue)


def intermed_plots(sess, feeds, model, visual, proc_queue):
    tmp_feed = feeds['eval_valid'].copy()
    tmp_feed[model.rnd_loc_eval] = True
    d = visual.eval_feed(sess, feed=tmp_feed, model=model)
    proc_queue.add_proc(target=visual.plot_planning_patches, args=(d, 2, '', 'rnd_loc_eval'), name='rnd_loc_eval')


def training_loop(FLAGS, sess, model, handles, writers, phase):
    feeds = model.get_feeds(FLAGS, handles)
    visual = Visualization_predRSSM(model, FLAGS)

    # if FLAGS.debug:
    #     max_processes = 0
    if (sys.platform == 'win32'):
        max_processes = 1
    else:
        max_processes = 4
    proc_queue = Proc_Queue(max_len=max_processes)

    if sess.run(model.global_step) == 0:
        run_eval(FLAGS, sess, feeds, model, writers, visual, proc_queue)

    for epochs_completed in range(phase['num_epochs']):
        # training
        t = time.time()
        train_op = model.get_train_op(FLAGS)

        for i in range(FLAGS.train_batches_per_epoch):
            if i and (i % 100 == 0):
                eval_stats = {'step': model.global_step, 'summary': model.summary, 'loss': model.loss, 'acc': model.acc, 'T': model.avg_T, 'acc_kn': model.acc_uk, 'acc_uk': model.acc_kn, 'G': model.avg_G}
                out_train = sess.run(eval_stats, feed_dict=feeds['eval_train'])
                out_valid = sess.run(eval_stats, feed_dict=feeds['eval_valid'])
                writers['train'].add_summary(out_train.pop('summary'), global_step=out_train.pop('step'))
                writers['valid'].add_summary(out_valid.pop('summary'), global_step=out_valid.pop('step'))
                # out_train.pop('G')
                # print(out_valid.pop('G'))
                stats = ['{}: {:.3f}'.format(k, v) for k, v in sorted(out_valid.items()) if not hasattr(v, "__len__")]
                print('{}/{}, '.format(i, FLAGS.train_batches_per_epoch) + ' '.join(stats))
                # # train
                # # print(out_train.pop('G'))
                # stats = ['{}: {:.3f}'.format(k, v) for k, v in sorted(out_train.items()) if not hasattr(v, "__len__")]
                # print('{}/{}, '.format(i, FLAGS.train_batches_per_epoch) + ' '.join(stats))

                if (FLAGS.planner == 'ActInf'):
                    intermed_plots(sess, feeds, model, visual, proc_queue)
            else:
                sess.run(train_op, feed_dict=feeds['train'])

        logger.info("{} epoch {}: {:.2f}min, {:.3f}s per batch, train op: {}".format(phase['name'], epochs_completed,
                                                                                      (time.time() - t) / 60,
                                                                                      (time.time() - t) / FLAGS.train_batches_per_epoch,
                                                                                      train_op.name))
        # evaluation
        if (epochs_completed % FLAGS.eval_step_interval) == 0:
            run_eval(FLAGS, sess, feeds, model, writers, visual, proc_queue)

    if phase['final_eval']:
        logger.info('FINISHED TRAINING, {} EPOCHS COMPLETED\n'.format(phase['num_epochs']))
        d = visual.eval_feed(sess, feeds['eval_test'], model)
        proc_queue.add_proc(target=visual.visualise, args=(d, '_test'), name='visualise_test')
        evaluate(FLAGS, sess, model, feeds['eval_test'],  FLAGS.batches_per_eval_test,  writers['test'], visual, proc_queue)

    logger.info('Cleaning up visualisation processes')
    proc_queue.cleanup()


def run_phase(FLAGS, phase, initial_phase, config, writers, train_data, valid_data, test_data):
    logger.info('Starting phase {}.'.format(phase['name']))
    tf.keras.backend.clear_session()

    cp_path = FLAGS.path + "/cp.ckpt"

    if FLAGS.uk_label and (phase['incl_uk'] == 0):
        x, y = train_data
        is_uk = (y == FLAGS.uk_label)
        train_data = (x[~is_uk], y[~is_uk])
        FLAGS.train_data_shape = (train_data[0].shape, train_data[1].shape)  # re-adjust

    with tf.Session(config=config) as sess:
        # Initialise env and model
        env = ImageForagingEnvironment(FLAGS)
        handles = env.intialise(train_data, valid_data, test_data, sess)
        model = predRSSM(FLAGS, env, phase)

        # Weights: initialise / restore from previous phase
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if initial_phase:
            writers['train'].add_graph(tf.get_default_graph())
        else:
            old_vars = [v[0] for v in tf.train.list_variables(cp_path)]
            variables_can_be_restored = [v for v in tf.global_variables() if v.name.split(':')[0] in old_vars]
            # [print(v) for v in variables_can_be_restored]
            # print('Unrestored:')
            # [print(v) for v in tf.global_variables() if v.name.split(':')[0] not in old_vars]

            # print(variables_can_be_restored)
            tf.train.Saver(variables_can_be_restored).restore(sess, cp_path)

        # Train
        training_loop(FLAGS, sess, model, handles, writers, phase)
        model.saver.save(sess, cp_path)

        return writers
