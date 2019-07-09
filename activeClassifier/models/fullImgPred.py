import warnings
import numpy as np
import tensorflow as tf
import logging
logger = logging.getLogger(__name__)

from activeClassifier.models import base
from activeClassifier.modules.policyNetwork import PolicyNetwork
from activeClassifier.modules_fullImg.generator import Generator
from activeClassifier.modules_fullImg.representation import Representation
from activeClassifier.modules_fullImg.planner import ActInfPlanner_fullImg, RandomLocPlanner
from activeClassifier.modules_fullImg.stateTransition import StateTransitionAdditive


class fullImgPred(base.Base):
    def __init__(self, FLAGS, env, phase):
        super().__init__(FLAGS, env, phase, name='fullImgPred')
        min_glimpses = 3
        policy = phase['policy']

        self.rnd_loc_eval = tf.placeholder_with_default(False, shape=(), name='rnd_loc_eval')

        if FLAGS.uk_cycling:  # randomly select a number of known class and mask them as FLAGS.uk_label
            # TODO: multinomial can draw the same class multiple times in the same draw
            p = 1. / FLAGS.num_classes_kn
            current_cycl_uk = tf.cond(self.is_training,
                                      lambda: tf.multinomial(tf.log(tf.fill([1, FLAGS.num_classes_kn], p)), num_samples=FLAGS.uk_cycling, output_dtype=tf.int32),  # [1, uks]
                                      lambda: tf.cast(tf.fill([1, FLAGS.num_classes_kn], -1), dtype=tf.int32),
                                      name='cycling_cond')

            y_exp = self.y_MC[:, tf.newaxis]
            is_cycling_uk = tf.reduce_any(tf.equal(y_exp, current_cycl_uk), axis=1)  # broadcasting to [B, uks], reducing to [B]
            self.y_MC = tf.where(is_cycling_uk, tf.fill([self.B], FLAGS.uk_label), self.y_MC)
            current_cycl_uk = tf.squeeze(current_cycl_uk, 0)  # [uks]
        else:
            current_cycl_uk = None

        # Initialise modules
        # n_policies = FLAGS.num_classes_kn if FLAGS.planner == 'ActInf' else 1
        policyNet = PolicyNetwork(FLAGS, self.B)
        reprNet   = Representation(FLAGS)
        generatorNet = Generator(FLAGS, batch_sz=self.B, input_shape=reprNet.output_shape, y_MC=env.y_MC)
        # planner = ActInfPlanner_fullImg(FLAGS, self.B, self.C)
        planner = RandomLocPlanner(FLAGS, self.B)
        stateTransition = StateTransitionAdditive(FLAGS, self.B, reprNet)

        fc_baseline = tf.layers.Dense(10, name='fc_baseline') if FLAGS.rl_reward == 'G' else tf.layers.Dense(1, name='fc_baseline')


        # self.n_policies = planner.n_policies

        # variables to remember. Probably to be implemented via TensorArray
        out_ta = []
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='reconstr'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='nll'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='KLdivs'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='G'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='actions'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='current_s'))
        out_ta.append(tf.TensorArray(tf.int32,   size=FLAGS.num_glimpses, dynamic_size=False, name='decisions'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='rewards'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='baselines'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses+1, dynamic_size=False, name='current_c'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses+1, dynamic_size=False, name='uk_belief'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='fb'))
        # out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='bl_surprise'))
        out_ta.append(tf.TensorArray(tf.bool,    size=FLAGS.num_glimpses, dynamic_size=False, name='done'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='exp_exp_obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='H_exp_exp_obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='exp_H'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='belief_loss'))

        ta_d = {}
        for i, ta in enumerate(out_ta):
            ta_d[ta.handle.name.split('/')[-1].replace(':0', '')] = ta

        # Initial values
        last_done = tf.zeros([self.B], dtype=tf.bool)
        last_decision = tf.fill([self.B], -1)
        last_state = stateTransition.initial_state
        generatorZero = generatorNet.zero_state
        prior_h, prior_z = generatorZero['h'], generatorZero['z']

        with tf.name_scope('Main_loop'):
            for time in range(FLAGS.num_glimpses):
                # expected scene
                VAE_results, next_exp_obs = generatorNet.class_cond_predictions(last_state['s'], prior_h, prior_z, env.img_NHWC, last_state['seen'], last_state['c'])

                next_decision, next_action, pl_records = planner.planning_step(last_state, next_exp_obs, time)

                bl_inputs = tf.concat([last_state['c'], tf.layers.flatten(last_state['s'])], axis=1)
                baseline = fc_baseline(tf.stop_gradient(bl_inputs))
                if FLAGS.rl_reward != 'G':
                    baseline = tf.squeeze(baseline, 1)

                observation, glimpse_idx, corr_classification_fb, done = env.step(next_action, next_decision)
                newly_done = done
                done = tf.logical_or(last_done, done)
                current_state = stateTransition(observation, next_action, last_state, VAE_results['KLdiv'], time, newly_done)
                # t=0 to T-1
                ta_d['obs']                      = self._write_zero_out(time, ta_d['obs'], observation, done, 'obs')
                ta_d['reconstr']                 = self._write_zero_out(time, ta_d['reconstr'], next_exp_obs['mu_probs'], done, 'reconstr')  # for visualisation only
                ta_d['actions']                  = self._write_zero_out(time, ta_d['actions'], next_action, done, 'actions')  # location actions, not including the decision acions
                # FOR BERNOULLI THIS HAS TO BE THE SAMPLE (MEAN IS THE UN-TRANSFORMED LOGITS), BUT FOR NORMAL DIST MIGHT WANT TO USE THE MEAN INSTEAD
                ta_d['current_s']                = self._write_zero_out(time, ta_d['current_s'], current_state['s'], done, 'current_s')  # [B, rnn]
                ta_d['fb']                       = self._write_zero_out(time, ta_d['fb'], current_state['fb'], done, 'fb')  # [B, num_classes]
                # ta_d['bl_surprise']              = self._write_zero_out(time, ta_d['bl_surprise'], bl_surprise, done, 'bl_surprise')  # [B]
                ta_d['done']                     = ta_d['done'].write(time, done)
                # t=0/1 to T
                ta_d['baselines']                = self._write_zero_out(time, ta_d['baselines'], baseline, last_done, 'baselines')  # this baseline is taken before the decision/observation! Same indexed rewards are taken after!
                ta_d['rewards']                  = self._write_zero_out(time, ta_d['rewards'], corr_classification_fb, last_done, 'rewards')
                # ta_d['belief_loss']              = self._write_zero_out(time, ta_d['belief_loss'], belief_loss, last_done, 'belief_loss')
                for k, v in pl_records.items():
                    if FLAGS.debug: print(time, k, v.shape)
                    ta_d[k] = self._write_zero_out(time, ta_d[k], v, done, k)
                # t=0 to T
                ta_d['current_c'] = self._write_zero_out(time + 1, ta_d['current_c'], current_state['c'], done, 'current_c')  # ONLY ONE t=0 TO T
                ta_d['uk_belief'] = self._write_zero_out(time + 1, ta_d['uk_belief'], current_state['uk_belief'], done, 'uk_belief')
                # TODO: SHOULD THESE BE done INSTEAD OF last_done??
                ta_d['KLdivs'] = self._write_zero_out(time, ta_d['KLdivs'], VAE_results['KLdiv'], last_done, 'KLdivs')  # [B, hyp]
                ta_d['nll'] = self._write_zero_out(time, ta_d['nll'], VAE_results['nll'], last_done, 'nll')  # [B, n_policies]
                # copy forward
                classification_decision = tf.where(last_done, last_decision, next_decision)
                last_decision = tf.where(last_done, last_decision, next_decision)
                ta_d['decisions'] = ta_d['decisions'].write(time, next_decision)
                # pass on to next time step
                last_done = done
                prior_h = VAE_results['h']
                prior_z = VAE_results['z']
                last_state = current_state

                # TODO: break loop if tf.reduce_all(last_done) (requires tf.while loop)
                time += 1

        with tf.name_scope('Stacking'):
            self.obs = ta_d['obs'].stack()  # [T,B,glimpse]
            self.actions = ta_d['actions'].stack()  # [T,B,2]
            self.decisions = ta_d['decisions'].stack()
            rewards = ta_d['rewards'].stack()
            done = ta_d['done'].stack()
            nll = ta_d['nll'].stack()  # [T,B,hyp]
            KLdiv_sum = ta_d['KLdivs'].stack()  # [T,B,hyp]
            ##########
            # LOSS IS ONLY INTERESTED IN POSTERIOR VALUES. POSTERIOR IS ALWAYS given the next glimpse, i.e. t+1 -> shift by -1 and append with zeros for the decision timestep
            ##########
            self.nll = tf.concat([nll[1:], tf.zeros_like(nll[0])[tf.newaxis]], axis=0)
            self.KLdivs = tf.concat([KLdiv_sum[1:], tf.zeros_like(KLdiv_sum[0])[tf.newaxis]], axis=0)

            self.state_believes = ta_d['current_c'].stack()  # [T+1,B,hyp]
            self.G = ta_d['G'].stack()  # [T,B,n_policies incl. correct/wrong fb]
            bl_loc = ta_d['baselines'].stack()
            self.reconstr = ta_d['reconstr'].stack()  # [T,B,hyp,glimpse]
            current_s = ta_d['current_s'].stack()  # [T,B,rnn]
            self.fb = ta_d['fb'].stack()  # [T,B,hyp]
            # bl_surprise = ta_d['bl_surprise'].stack()  # [T,B]
            self.uk_belief = ta_d['uk_belief'].stack()  # [T+1,B]
            # belief_loss = ta_d['belief_loss'].stack()  # [T,B]

            # further records for debugging
            self.exp_exp_obs = ta_d['exp_exp_obs'].stack()  # [T, B, num_classes_kn, z]
            self.H_exp_exp_obs = ta_d['H_exp_exp_obs'].stack()
            self.exp_H = ta_d['exp_H'].stack()

            self.num_glimpses_dyn = tf.shape(self.obs)[0]
            T = tf.cast(self.num_glimpses_dyn, tf.float32) - tf.count_nonzero(done, 0, dtype=tf.float32)  # NOTE: does include the decision action, i.e. T-1 glimpses taken, then decided
            self.avg_T = tf.reduce_mean(T)

            # if FLAGS.pixel_obs_discrete:
            #     self.reconstr_posterior = self._pixel_obs_discrete_into_glimpse(self.reconstr_posterior, FLAGS)
            #     self.exp_exp_obs = self._pixel_obs_discrete_into_glimpse(self.exp_exp_obs, FLAGS)

        with tf.name_scope('Losses'):
            with tf.name_scope('RL'):
                returns = tf.cumsum(rewards, reverse=True, axis=0)
                # baseline is calculated before taking a new obs, rewards with same index thereafter
                baseline_mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(returns) - bl_loc), axis=0))

            with tf.name_scope('Classification'):
                # might never make a classification decision
                # TODO: SHOULD I FORCE THE ACTION AT t=t TO BE A CLASSIFICATION?
                self.classification = classification_decision

            with tf.name_scope('VAE'):
                correct_hypoths = tf.cast(tf.one_hot(env.y_MC, depth=FLAGS.num_classes_kn), tf.bool)  # [B, hyp]

                # posterior nll
                nll_post_sum_T = tf.reduce_sum(self.nll, 0)  # sum over time, [B, hyp]
                # NOTE: nll masking moved into generator
                # TODO: meaning that nll_loss displayed in visualisation is wrong
                # nll_post_used = tf.boolean_mask(nll_post_sum_T, mask=correct_hypoths, axis=0)  # [?]
                # nll_post = tf.reduce_mean(nll_post_used)  # mean over batch
                nll_post = tf.reduce_mean(nll_post_sum_T)  # mean over batch

                # KLdiv: train model to predict glimpses where (i) the hyp is correct (ii) the location has previously been seen
                incl_seen = False
                if not incl_seen:
                    KLdivs_used = tf.boolean_mask(self.KLdivs, mask=correct_hypoths, axis=1)  # [T, ?]
                else:
                    start_from_epoch = 0  # at beginning all locations are 0, give it some time to develop strategie that don't look at same loc all the time
                    closeness = 3  # in pixel
                    # seen_locs = tf.cond(self.epoch_num >= start_from_epoch,
                    #                     lambda: policyNet.find_seen_locs(self.actions, FLAGS.num_glimpses, closeness=closeness),
                    #                     lambda: tf.zeros([FLAGS.num_glimpses, self.B], dtype=tf.bool),
                    #                     name='seen_locs_cond')
                    # if FLAGS.uk_label is not None:
                    #     seen_locs = tf.logical_and(seen_locs, tf.not_equal(self.y_MC, FLAGS.uk_label)[tf.newaxis, :])
                    #
                    # mask = tf.logical_or(seen_locs[:, :, tf.newaxis], correct_hypoths[tf.newaxis, :, :])  # broadcast into [T, B, hyp]
                    # # KLdivs_used = tf.boolean_mask(self.KLdivs, mask=mask)  # not shape preserving: [?]
                    # KLdivs_used = tf.where(mask, self.KLdivs, tf.zeros_like(self.KLdivs))
                self.KLdiv_sum = tf.reduce_sum(KLdivs_used) / tf.cast(self.B, tf.float32)  # sum over time

            with tf.name_scope('Bl_surprise'):
                if FLAGS.normalise_fb:
                    pass
                    # if FLAGS.uk_label is not None:
                        # bl_surprise = tf.boolean_mask(bl_surprise, mask=tf.not_equal(self.y_MC, FLAGS.uk_label), axis=1)
                    # bl_surpise_mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(KLdivs_used) - tf.squeeze(bl_surprise, 2)), axis=0))
                else:
                    bl_surpise_mse = tf.constant(0.)

            # beliefUpdate_loss = tf.reduce_mean(belief_loss)
            beliefUpdate_loss = 0.

            ctrls = []
            if FLAGS.debug:
                for var, name in [(baseline_mse, 'baseline_mse'), (beliefUpdate_loss, 'beliefUpdate_loss'),
                                  (bl_surpise_mse, 'bl_surpise_mse'), (nll_post, 'nll_post'), (self.KLdiv_sum, 'KLdiv')]:
                    non_nan = tf.logical_not(tf.reduce_any(tf.is_nan(var)))
                    ctrls.append(tf.assert_equal(non_nan, True, name='isnan_{}'.format(name)))
            # TODO: SCALE LOSSES DIFFERENTLY? (only necessary if they flow into the same weights, might not be the case so far)
            with tf.control_dependencies(ctrls):
                self.loss = baseline_mse + beliefUpdate_loss + bl_surpise_mse + nll_post + self.KLdiv_sum

        with tf.variable_scope('Optimizer'):
            def drop_vars(collection, to_drop):
                return list(set(collection) - set(to_drop))

            self.train_op, _ = self._create_train_op(FLAGS, self.loss, self.global_step, name='train_op_full')

            all_vars       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            excl_enc       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='/'.join([reprNet.name, 'posterior']))
            assert (excl_enc is not None)
            self.train_op_freezeEncoder, _ = self._create_train_op(FLAGS, self.loss, self.global_step, varlist=drop_vars(all_vars, excl_enc), name='train_op_freezeEncoder')

        with tf.name_scope('Monitoring'):
            tf.summary.scalar('lr', self.learning_rate)
            tf.summary.scalar('loc_std', policyNet.std)

        with tf.name_scope('Summaries'):
            metrics_upd_coll = "streaming_updates"

            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y_MC, classification_decision), tf.float32))  # only to get easy direct intermendiate outputs
            self.avg_G = tf.reduce_mean(self.G, axis=[1])  # [T, n_policies]

            fb_ratio_bestWrong_corr_ts_seen = []
            fb_ratio_bestWrong_corr_ts_unseen = []
            for t in range(FLAGS.num_glimpses):
                fb_t = self.fb[t]  # [B, hyp]
                # TODO: might be wrong for uk, assuming there is always a correct hyp per obs
                fb_t_corr = tf.reduce_max(tf.where(correct_hypoths, fb_t, tf.zeros_like(fb_t)), axis=1)
                fb_t_bestWrong = tf.reduce_min(tf.where(correct_hypoths, tf.ones_like(fb_t) * 1e4, fb_t), axis=1)
                # fb_t_wrong, fb_t_corr = tf.dynamic_partition(fb_t, correct_hypoths, num_partitions=2)

                ratio = fb_t_bestWrong / fb_t_corr

                if incl_seen:
                    ratio_unseen, ratio_seen = tf.dynamic_partition(ratio, tf.cast(seen_locs[t], tf.int32), num_partitions=2)
                    fb_ratio_bestWrong_corr_ts_unseen.append(tf.reduce_mean(ratio_unseen))
                    fb_ratio_bestWrong_corr_ts_seen.append(tf.reduce_mean(ratio_seen))
                else:
                    fb_ratio_bestWrong_corr_ts_unseen.append(tf.reduce_mean(ratio))

            scalars = {'Main/loss': self.loss,
                       'Main/acc': self.acc,
                       'loss/VAE_nll_post': nll_post,
                       'loss/VAE_KLdiv': self.KLdiv_sum,
                       'loss/RL_locBl_mse': tf.reduce_mean(baseline_mse),
                       'loss/RL_returns': tf.reduce_mean(returns),
                       'loss/BU_loss': beliefUpdate_loss,
                       'loss/BU_surpiseBL_mse': bl_surpise_mse,
                       'misc/pct_noDecision': tf.count_nonzero(tf.equal(classification_decision, -1), dtype=tf.float32) / tf.cast(self.B, tf.float32),
                       'misc/T': self.avg_T,
                       }
            for t in range(FLAGS.num_glimpses):
                scalars['misc/fb_ratio_bestWrong_corr_t{}_unseen'.format(t)] = fb_ratio_bestWrong_corr_ts_unseen[t]
                if incl_seen:
                    scalars['misc/fb_ratio_bestWrong_corr_t{}_seen'.format(t)] = fb_ratio_bestWrong_corr_ts_seen[t]

                # would expect this to get better over time as predictions are built upon more information (min over hypotheses). Though might be influenced by loc policy if not random
                scalars['misc/bestKLdiv_t{}'.format(t)] = tf.reduce_mean(tf.reduce_min(self.KLdivs[t], axis=-1))

            if FLAGS.uk_label:
                corr = tf.equal(self.y_MC, classification_decision)
                is_uk = tf.equal(self.y_MC, FLAGS.uk_label)
                corr_kn, corr_uk = tf.dynamic_partition(corr, partitions=tf.cast(is_uk, tf.int32), num_partitions=2)
                self.acc_kn = tf.reduce_mean(tf.cast(corr_kn, tf.float32))
                self.acc_uk = tf.reduce_mean(tf.cast(corr_uk, tf.float32))  # can be nan if there are no uks
                share_clf_uk = tf.reduce_mean(tf.cast(tf.equal(classification_decision, FLAGS.uk_label), tf.float32))
                scalars['uk/acc_kn'] = self.acc_kn
                scalars['uk/acc_uk'] = self.acc_uk
                scalars['uk/share_clf_uk'] = share_clf_uk
            else:
                self.acc_kn, self.acc_uk, share_clf_uk = tf.constant(0.), tf.constant(0.), tf.constant(0.)

            for name, scalar in scalars.items():
                tf.summary.scalar(name, scalar)
                tf.metrics.mean(scalar, name=name, updates_collections=metrics_upd_coll)

            self.metrics_update = tf.get_collection(metrics_upd_coll)
            self.metrics_names = [v.name.replace('_1/update_op:0', '').replace('Summaries/', '') for v in self.metrics_update]

        # histograms
        # self.z_post = ta_d['z_post'].stack()
        if FLAGS.debug:
            for t in range(FLAGS.num_glimpses):
                # tf.summary.histogram('t{}/z_post'.format(t), self.z_post[t])
                tf.summary.histogram('t{}/exp_exp_obs'.format(t), self.exp_exp_obs[t])
                tf.summary.histogram('t{}/sum_obs'.format(t), tf.reduce_sum(self.obs[t], axis=-1))

                # if self.n_policies == 1:
                #     if FLAGS.size_z == 10:
                #         shp = [5, 2]
                #     elif FLAGS.size_z == 32:
                #         shp = [8, 4]
                #     elif FLAGS.size_z == 128:
                #         shp = [16, 8]
                #     else:
                #         shp = 2 * [int(np.sqrt(FLAGS.size_z))]
                #         if np.prod(shp) != FLAGS.size_z:
                #             continue
                #     # TODO: selected_exp_obs_enc are per hyp, exp_exp_obs per policy
                #     shp = [self.B] + shp + [1]
                #     tf.summary.image('t{}/z_prior'.format(t), tf.reshape(selected_exp_obs_enc[t], shp), max_outputs=3)
                #     tf.summary.image('t{}/z_post'.format(t), tf.reshape(self.z_post[t], shp), max_outputs=3)
                #     tf.summary.image('t{}/exp_exp_obs'.format(t), tf.reshape(self.exp_exp_obs[t], shp), max_outputs=3)
                #     tf.summary.image('t{}/obs'.format(t), tf.reshape(self.obs[t], [self.B, 8, 8, 1]), max_outputs=3)

        self.summary = tf.summary.merge_all()

        with tf.name_scope('visualisation'):
            self.glimpses_composed = env.composed_glimpse(FLAGS, self.obs, self.num_glimpses_dyn)

        self.saver = self._create_saver(phase)


    def get_train_op(self, FLAGS):
        if (FLAGS.freeze_enc is not None) and (FLAGS.freeze_policyNet is not None):
            warnings.warn('BOTH_FREEZE_VALUES_ARE_NOT_NONE_IS_THIS_WANTED?')
        epoch = self.epoch_num.eval()
        if (FLAGS.freeze_enc is not None) and (epoch >= FLAGS.freeze_enc):
            return self.train_op_freezeEncoder
        else:
            return self.train_op


    def _pixel_obs_discrete_into_glimpse(self, reconstr, FLAGS):
        expectation = reconstr * tf.range(FLAGS.pixel_obs_discrete, dtype=tf.float32)
        expectation /= FLAGS.pixel_obs_discrete  # [0-1] range
        return tf.reduce_mean(expectation, axis=-1)
        # return tf.argmax(reconstr, axis=-1, output_type=tf.float32) / FLAGS.pixel_obs_discrete


    def get_visualisation_fetch(self):
        return {'nll'      : self.nll,
                'reconstr' : self.reconstr,
                }