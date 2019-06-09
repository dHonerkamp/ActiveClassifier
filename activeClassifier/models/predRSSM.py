import warnings
import tensorflow as tf

from activeClassifier.models import base
from activeClassifier.tools.tf_tools import repeat_axis
from activeClassifier.modules.policyNetwork import PolicyNetwork
from activeClassifier.modules.VAE.encoder import Encoder, EncoderConv
from activeClassifier.modules.VAE.decoder import Decoder
from activeClassifier.modules.planner.ActInfPlanner import ActInfPlanner
from activeClassifier.modules.planner.REINFORCEPlanner import REINFORCEPlanner
from activeClassifier.modules.stateTransition import StateTransition
from activeClassifier.modules.beliefUpdate import PredErrorUpdate, FullyConnectedUpdate, RAMUpdate


class predRSSM(base.Base):
    def __init__(self, FLAGS, env, phase):
        super().__init__(FLAGS, env, phase)
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
        n_policies = FLAGS.num_classes_kn if FLAGS.planner == 'ActInf' else 1
        policyNet = PolicyNetwork(FLAGS, self.B)
        if FLAGS.convLSTM:
            VAEencoder   = EncoderConv(FLAGS, env.patch_shape, self.is_training)
        else:
            VAEencoder   = Encoder(FLAGS, env.patch_shape, self.is_training)
        VAEdecoder   = Decoder(FLAGS, env.patch_shape_flat)
        stateTransition = StateTransition(FLAGS, FLAGS.size_rnn)
        fc_baseline = tf.layers.Dense(10, name='fc_baseline') if FLAGS.rl_reward == 'G' else tf.layers.Dense(1, name='fc_baseline')

        submodules = {'policyNet': policyNet,
                      'VAEEncoder': VAEencoder,
                      'VAEDecoder': VAEdecoder}

        planner = policy if policy != 'random' else FLAGS.planner
        if planner == 'ActInf':
            planner = ActInfPlanner(FLAGS, submodules, self.B, env.patch_shape_flat, stateTransition, self.C)
        elif planner == 'RL':
            planner = REINFORCEPlanner(FLAGS, submodules, self.B, env.patch_shape_flat, stateTransition, is_pre_phase=(FLAGS.planner!='RL'), labels=self.y_MC)
        else:
            raise ValueError('Undefined planner.')

        self.n_policies = planner.n_policies
        if FLAGS.beliefUpdate == 'fb':
            beliefUpdate = PredErrorUpdate(FLAGS, submodules, self.B, labels=self.y_MC, current_cycl_uk=current_cycl_uk)
        elif FLAGS.beliefUpdate == 'fc':
            beliefUpdate = FullyConnectedUpdate(FLAGS, submodules, self.B, labels=self.y_MC, current_cycl_uk=current_cycl_uk)
        elif FLAGS.beliefUpdate == 'RAM':
            beliefUpdate = RAMUpdate(FLAGS, submodules, self.B, labels=self.y_MC, current_cycl_uk=current_cycl_uk)
        else:
            raise ValueError('Undefined beliefUpdate.')

        # variables to remember. Probably to be implemented via TensorArray
        out_ta = []
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='nll_posterior'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='reconstr_posterior'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='KLdivs'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='G'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='actions'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='actions_mean'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='selected_exp_obs_enc'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='current_s'))
        out_ta.append(tf.TensorArray(tf.int32,   size=FLAGS.num_glimpses, dynamic_size=False, name='decisions'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='rewards'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='baselines'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses+1, dynamic_size=False, name='current_c'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='uk_belief'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='fb'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='bl_surprise'))
        out_ta.append(tf.TensorArray(tf.bool,    size=FLAGS.num_glimpses, dynamic_size=False, name='done'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='exp_exp_obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='exp_obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='H_exp_exp_obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='exp_H'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='potential_actions'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='belief_loss'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='z_post'))
        out_ta.append(tf.TensorArray(tf.int32, size=FLAGS.num_glimpses, dynamic_size=False, name='selected_action_idx'))
        out_ta.append(tf.TensorArray(tf.float32, size=FLAGS.num_glimpses, dynamic_size=False, name='rewards_Gobs'))

        ta_d = {}
        for i, ta in enumerate(out_ta):
            ta_d[ta.handle.name.split('/')[-1].replace(':0', '')] = ta

        # Initial values
        last_done = tf.zeros([self.B], dtype=tf.bool)
        last_decision = tf.fill([self.B], -1)
        if not FLAGS.rnd_first_glimpse:
            last_state = stateTransition.initial_state(self.B, tf.zeros([self.B, policyNet.output_size]))
            last_z = tf.zeros([self.B] + VAEencoder.output_shape)

        with tf.name_scope('Main_loop'):
            for time in range(FLAGS.num_glimpses):
                if FLAGS.rnd_first_glimpse and (time == 0):
                    # Initial action (if model can plan first action, the optimal first location should be identical for all images)
                    current_state, next_decision, next_action, next_action_mean, next_exp_obs, pl_records = planner.initial_planning()
                    ta_d['current_c'] = self._write_zero_out(0, ta_d['current_c'], current_state['c'], last_done, 'current_c')
                else:
                    if policy == 'random':
                        next_decision, next_action, next_action_mean, pl_records = planner.random_policy(FLAGS.init_loc_rng)
                        next_exp_obs = planner.single_policy_prediction(last_state, last_z, next_action)
                    else:
                        next_decision, next_action, next_action_mean, next_exp_obs, pl_records = planner.planning_step(last_state, last_z, time, self.is_training, self.rnd_loc_eval)

                    # TODO : Could REUSE FROM PLANNING STEP
                    current_state = stateTransition([last_z, next_action], last_state)

                bl_inputs = tf.concat([current_state['c'], current_state['s']], axis=1)
                baseline = fc_baseline(tf.stop_gradient(bl_inputs))
                if FLAGS.rl_reward != 'G':
                    baseline = tf.squeeze(baseline, 1)

                observation, corr_classification_fb, done = env.step(next_action, next_decision)
                newly_done = done
                done = tf.logical_or(last_done, done)
                current_state, z_samples, nll_post, reconstr_posterior, KLdivs, belief_loss, bl_surprise = beliefUpdate.update(current_state, observation, next_exp_obs, time, newly_done)
                # t=0 to T-1
                ta_d['obs']                      = self._write_zero_out(time, ta_d['obs'], observation, done, 'obs')
                ta_d['KLdivs']                   = self._write_zero_out(time, ta_d['KLdivs'], KLdivs, done, 'KLdivs')  # [B, hyp]
                ta_d['nll_posterior']            = self._write_zero_out(time, ta_d['nll_posterior'], nll_post, done, 'nll_posterior')  # [B, n_policies]
                ta_d['reconstr_posterior']       = self._write_zero_out(time, ta_d['reconstr_posterior'], reconstr_posterior, done, 'reconstr_posterior')  # for visualisation only
                ta_d['actions']                  = self._write_zero_out(time, ta_d['actions'], next_action, done, 'actions')  # location actions, not including the decision acions
                ta_d['actions_mean']             = self._write_zero_out(time, ta_d['actions_mean'], next_action_mean, done, 'actions_mean')  # location actions, not including the decision acions
                # FOR BERNOULLI THIS HAS TO BE THE SAMPLE (MEAN IS THE UN-TRANSFORMED LOGITS), BUT FOR NORMAL DIST MIGHT WANT TO USE THE MEAN INSTEAD
                ta_d['selected_exp_obs_enc']     = self._write_zero_out(time, ta_d['selected_exp_obs_enc'], next_exp_obs['sample'], done, 'selected_exp_obs_enc')
                ta_d['current_s']                = self._write_zero_out(time, ta_d['current_s'], current_state['s'], done, 'current_s')  # [B, rnn]
                ta_d['fb']                       = self._write_zero_out(time, ta_d['fb'], current_state['fb'], done, 'fb')  # [B, num_classes]
                ta_d['bl_surprise']              = self._write_zero_out(time, ta_d['bl_surprise'], bl_surprise, done, 'bl_surprise')  # [B]
                ta_d['done']                     = ta_d['done'].write(time, done)
                # t=0/1 to T
                ta_d['baselines']                = self._write_zero_out(time, ta_d['baselines'], baseline, last_done, 'baselines')  # this baseline is taken before the decision/observation! Same indexed rewards are taken after!
                ta_d['rewards']                  = self._write_zero_out(time, ta_d['rewards'], corr_classification_fb, last_done, 'rewards')
                ta_d['belief_loss']              = self._write_zero_out(time, ta_d['belief_loss'], belief_loss, last_done, 'belief_loss')
                ta_d['z_post']                   = self._write_zero_out(time, ta_d['z_post'], z_samples, last_done, 'z_post')
                # t=0 to T
                for k, v in pl_records.items():
                    if FLAGS.debug: print(time, k, v.shape)
                    ta_d[k] = self._write_zero_out(time, ta_d[k], v, done, k)
                ta_d['current_c'] = self._write_zero_out(time + 1, ta_d['current_c'], current_state['c'], last_done, 'current_c')  # ONLY ONE t=0 TO T
                ta_d['uk_belief'] = self._write_zero_out(time, ta_d['uk_belief'], current_state['uk_belief'], last_done, 'uk_belief')
                # copy forward
                classification_decision = tf.where(last_done, last_decision, next_decision)
                last_decision = tf.where(last_done, last_decision, next_decision)
                ta_d['decisions'] = ta_d['decisions'].write(time, next_decision)
                # pass on to next time step
                last_done = done
                last_z = z_samples
                # last_c = current_c  # TODO: could also use the one from planning (new_c) or pi
                last_state = current_state

                # TODO: break loop if tf.reduce_all(last_done) (requires tf.while loop)
                time += 1

        with tf.name_scope('Stacking'):
            self.obs = ta_d['obs'].stack()  # [T,B,glimpse]
            self.actions = ta_d['actions'].stack()  # [T,B,2]
            actions_mean = ta_d['actions_mean'].stack()  # [T,B,2]
            self.decisions = ta_d['decisions'].stack()
            rewards = ta_d['rewards'].stack()
            done = ta_d['done'].stack()
            self.nll_posterior = ta_d['nll_posterior'].stack()  # [T,B,hyp]
            self.KLdivs = ta_d['KLdivs'].stack()  # [T,B,hyp]
            self.state_believes = ta_d['current_c'].stack()  # [T+1,B,hyp]
            self.G = ta_d['G'].stack()  # [T,B,n_policies incl. correct/wrong fb]
            bl_loc = ta_d['baselines'].stack()
            self.reconstr_posterior = ta_d['reconstr_posterior'].stack()  # [T,B,hyp,glimpse]
            current_s = ta_d['current_s'].stack()  # [T,B,rnn]
            self.fb = ta_d['fb'].stack()  # [T,B,hyp]
            bl_surprise = ta_d['bl_surprise'].stack()  # [T,B]
            self.uk_belief = ta_d['uk_belief'].stack()  # [T,B]
            belief_loss = ta_d['belief_loss'].stack()  # [T,B]

            # further records for debugging
            self.selected_exp_obs_enc = ta_d['selected_exp_obs_enc'].stack()
            self.exp_obs = ta_d['exp_obs'].stack()  # [T, B, n_policies, num_classes_kn, z]
            self.exp_exp_obs = ta_d['exp_exp_obs'].stack()  # [T, B, num_classes_kn, z]
            self.H_exp_exp_obs = ta_d['H_exp_exp_obs'].stack()
            self.exp_H = ta_d['exp_H'].stack()
            self.potential_actions = ta_d['potential_actions'].stack()  # [T,B,n_policies,loc]
            self.selected_action_idx = ta_d['selected_action_idx'].stack()  # [T, B]

            self.num_glimpses_dyn = tf.shape(self.obs)[0]
            T = tf.cast(self.num_glimpses_dyn, tf.float32) - tf.count_nonzero(done, 0, dtype=tf.float32)  # NOTE: does include the decision action, i.e. T-1 glimpses taken, then decided
            self.avg_T = tf.reduce_mean(T)

            if FLAGS.pixel_obs_discrete:
                self.reconstr_posterior = self._pixel_obs_discrete_into_glimpse(self.reconstr_posterior, FLAGS)
                self.exp_obs = self._pixel_obs_discrete_into_glimpse(self.exp_obs, FLAGS)
                self.exp_exp_obs = self._pixel_obs_discrete_into_glimpse(self.exp_exp_obs, FLAGS)

        with tf.name_scope('Losses'):
            with tf.name_scope('RL'):
                # TODO: WITH RL AS PRE-TRAINING THIS MEANS THE REWARDS ARE CHANGED WHEN MOVING TO MAIN PHASE
                if (FLAGS.planner == 'ActInf') and (FLAGS.rl_reward == 'G'):
                    returns = ta_d['rewards_Gobs'].stack()  # [T, B, n_policies]
                    # returns = tf.Print(returns, [returns[:, 0, :]], summarize=100)
                    policy_losses = policyNet.REINFORCE_losses(returns, bl_loc, self.actions[:, :, tf.newaxis], actions_mean[:, :, tf.newaxis])  # [T,B]
                else:
                    if (FLAGS.planner == 'ActInf') and (FLAGS.rl_reward == 'G1'):
                        assert planner.n_policies == 1
                        # TODO: use G before the prior preferences to remove noise from lower valued for more than 4 glimpses?
                        # returns = self.G[:, :, 0]  # excluding the decision action
                        returns = tf.squeeze(self.H_exp_exp_obs - self.exp_H, -1)
                        # returns = tf.squeeze(self.H_exp_exp_obs, -1)
                    else:
                        returns = tf.cumsum(rewards, reverse=True, axis=0)
                    policy_losses = policyNet.REINFORCE_losses(returns, bl_loc, self.actions, actions_mean)  # [T,B]
                policy_loss   = tf.reduce_mean(tf.reduce_sum(policy_losses, 0))  # sum over time
                # baseline is calculated before taking a new obs, rewards with same index thereafter
                baseline_mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(returns) - bl_loc), axis=0))

            with tf.name_scope('Classification'):
                # might never make a classification decision
                # TODO: SHOULD I FORCE THE ACTION AT t=t TO BE A CLASSIFICATION?
                self.classification = classification_decision

            with tf.name_scope('VAE'):
                # mask losses of wrong hyptheses
                nll_post = tf.reduce_sum(self.nll_posterior, 0)  # sum over time
                nll_post = tf.reduce_mean(nll_post)  # mean over batch

                correct_hypoths = tf.cast(tf.one_hot(env.y_MC, depth=FLAGS.num_classes_kn), tf.bool)
                KLdivs_correct = tf.boolean_mask(self.KLdivs, mask=correct_hypoths, axis=1)
                KLdiv = tf.reduce_mean(tf.reduce_sum(KLdivs_correct, axis=0))  # sum over time

            with tf.name_scope('Bl_surprise'):
                if FLAGS.normalise_fb:
                    if FLAGS.uk_label is not None:
                        bl_surprise = tf.boolean_mask(bl_surprise, mask=tf.not_equal(self.y_MC, FLAGS.uk_label), axis=1)
                    bl_surpise_mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(KLdivs_correct) - tf.squeeze(bl_surprise, 2)), axis=0))
                else:
                    bl_surpise_mse = tf.constant(0.)

            beliefUpdate_loss = tf.reduce_mean(belief_loss)

            ctrls = []
            if FLAGS.debug:
                for var, name in [(policy_loss, 'policy_loss'), (baseline_mse, 'baseline_mse'), (beliefUpdate_loss, 'beliefUpdate_loss'),
                                  (bl_surpise_mse, 'bl_surpise_mse'), (nll_post, 'nll_post'), (KLdiv, 'KLdiv')]:
                    ctrls.append(tf.logical_not(tf.reduce_any(tf.is_nan(var)), name='isnan_{}'.format(name)))
            # TODO: SCALE LOSSES DIFFERENTLY? (only necessary if they flow into the same weights, might not be the case so far)
            with tf.control_dependencies(ctrls):
                self.loss = policy_loss + baseline_mse + beliefUpdate_loss + bl_surpise_mse + nll_post + KLdiv

        with tf.variable_scope('Optimizer'):
            def drop_vars(collection, to_drop):
                return list(set(collection) - set(to_drop))

            if (policy == 'random') or FLAGS.uniform_loc10:  # don't train locationNet or location baseline
                pretrain_vars = (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=VAEencoder.name)
                                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=VAEdecoder.name)
                                 + stateTransition._cell.trainable_variables
                                 # + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=beliefUpdate.name)
                                 )
                print('Variables trained:')
                [print(v) for v in pretrain_vars]
                print()
                print('Variables excluded:')
                excl = drop_vars(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), pretrain_vars)
                [print (v) for v in excl]
                assert len(excl) == 2  # baseline bias and kernel, policyNet never gets initialised
                self.train_op, gradient_check_Pre, _ = self._create_train_op(FLAGS, self.loss, self.global_step, name='train_op_pretrain', varlist=pretrain_vars)
            else:
                self.train_op, gradient_check_F, _ = self._create_train_op(FLAGS, self.loss, self.global_step, name='train_op_full')

            all_vars       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            excl_enc       = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='/'.join([VAEencoder.name, 'posterior']))
            excl_policyNet = (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=policyNet.name)
                              + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_baseline'))
            assert (excl_enc is not None) and (excl_policyNet is not None)
            loss_exclPolicy = beliefUpdate_loss + bl_surpise_mse + nll_post + KLdiv
            self.train_op_freezeEncoder, _, _ = self._create_train_op(FLAGS, self.loss, self.global_step, varlist=drop_vars(all_vars, excl_enc), name='train_op_freezeEncoder')
            self.train_op_freezePolNet, _, _  = self._create_train_op(FLAGS, loss_exclPolicy, self.global_step, varlist=drop_vars(all_vars, excl_policyNet), name='train_op_freezePolNet')

        with tf.name_scope('Monitoring'):
            tf.summary.scalar('lr', self.learning_rate)
            tf.summary.scalar('loc_std', policyNet.std)

        with tf.name_scope('Summaries'):
            metrics_upd_coll = "streaming_updates"

            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y_MC, classification_decision), tf.float32))  # only to get easy direct intermendiate outputs
            self.avg_G = tf.reduce_mean(self.G, axis=[1])  # [T, n_policies]

            scalars = {'Main/loss': self.loss,
                       'Main/acc': self.acc,
                       'loss/VAE_nll_post': nll_post,
                       'loss/VAE_KLdiv': KLdiv,
                       'loss/RL_locBl_mse': tf.reduce_mean(baseline_mse),
                       'loss/RL_policyLoss': policy_loss,
                       'loss/RL_returns': tf.reduce_mean(returns),
                       'loss/BU_loss': beliefUpdate_loss,
                       'loss/BU_surpiseBL_mse': bl_surpise_mse,
                       'misc/pct_noDecision': tf.count_nonzero(tf.equal(classification_decision, -1), dtype=tf.float32) / tf.cast(self.B, tf.float32),
                       'misc/T': self.avg_T,
                       }
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
        self.z_post = ta_d['z_post'].stack()
        if FLAGS.debug:
            for t in range(FLAGS.num_glimpses):
                tf.summary.histogram('t{}/z_prior'.format(t), self.selected_exp_obs_enc[t])
                tf.summary.histogram('t{}/z_post'.format(t), self.z_post[t])
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
            # TODO: wouldn't have to do this if FLAGS.use_pixel_obs_FE (selected_exp_obs_enc are the z-exp obs, not the pixel level ones)
            def reshp_tile(x):
                x = tf.reshape(x, [FLAGS.num_glimpses * self.B, 1, x.shape[-1]])
                x = tf.tile(x, [1, FLAGS.num_classes_kn, 1])
                return tf.reshape(x, [FLAGS.num_glimpses * self.B * FLAGS.num_classes_kn, x.shape[-1]])

            reconstr_prior = VAEdecoder.decode([tf.reshape(self.selected_exp_obs_enc, [FLAGS.num_glimpses * self.B * FLAGS.num_classes_kn] + VAEencoder.output_shape),
                                                reshp_tile(self.actions)],
                                               true_glimpse=None,  # reshp_tile(self.obs), # only needed if also want the loss. Need to also one-hot encode if FLAGS.pixel_obs_discrete
                                               out_shp=[FLAGS.num_glimpses, self.B, FLAGS.num_classes_kn])
            # Can be non-zero for zeroed-out time steps due to the learned bias of  the VAE_decoder
            self.reconstr_prior = reconstr_prior['sample']
            if FLAGS.pixel_obs_discrete:
                self.reconstr_prior = self._pixel_obs_discrete_into_glimpse(self.reconstr_prior, FLAGS)

        self.saver = self._create_saver(phase)


    def get_train_op(self, FLAGS):
        if (FLAGS.freeze_enc is not None) and (FLAGS.freeze_policyNet is not None):
            warnings.warn('BOTH_FREEZE_VALUES_ARE_NOT_NONE_IS_THIS_WANTED?')
        epoch = self.epoch_num.eval()
        if (FLAGS.freeze_enc is not None) and (epoch >= FLAGS.freeze_enc):
            return self.train_op_freezeEncoder
        elif (FLAGS.freeze_policyNet is not None) and (epoch >= FLAGS.freeze_policyNet):
            return self.train_op_freezePolNet
        else:
            return self.train_op


    def _pixel_obs_discrete_into_glimpse(self, reconstr, FLAGS):
        expectation = reconstr * tf.range(FLAGS.pixel_obs_discrete, dtype=tf.float32)
        expectation /= FLAGS.pixel_obs_discrete  # [0-1] range
        return tf.reduce_mean(expectation, axis=-1)
        # return tf.argmax(reconstr, axis=-1, output_type=tf.float32) / FLAGS.pixel_obs_discrete


    def _write_zero_out(self, time, ta, candidate, done, name):
        if self.debug and (candidate.dtype == tf.float32):
            ctrl = [tf.logical_not(tf.reduce_any(tf.is_nan(candidate)), name='ctrl_{}'.format(name))]
        else:
            ctrl = []
        with tf.control_dependencies(ctrl):
            ta = ta.write(time, tf.where(done, tf.zeros_like(candidate), candidate))
        return ta