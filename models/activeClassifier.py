import tensorflow as tf
from tensorflow import distributions as tfd

from models import base
from tools.tf_tools import TINY, write_zero_out, repeat_axis, calculate_gaussian_nll
from modules.policyNetwork import PolicyNetwork
from modules.VAE import Encoder, Decoder, GlimpseEncoder
from modules.planner import ActInfPlanner, REINFORCEPlanner
from modules.stateTransition import StateTransition_AC


class ActiveClassifier(base.Base):
    def __init__(self, FLAGS, env, phase):
        super().__init__(FLAGS, env, phase)
        min_glimpses = 3
        random_locations = phase['random_locations']  # tf.logical_and(self.epoch_num < FLAGS.pre_train_epochs, self.is_training)

        # Initialise modules
        n_policies = FLAGS.num_classes if FLAGS.planner == 'ActInf' else 1
        policyNet = PolicyNetwork(FLAGS, self.B, n_policies)
        glimpseEncoder = GlimpseEncoder(FLAGS)
        VAEencoder   = Encoder(FLAGS, env.patch_shape_flat)
        VAEdecoder   = Decoder(FLAGS, env.patch_shape_flat)
        stateTransition_AC = StateTransition_AC(FLAGS.size_rnn, 2*FLAGS.size_z)
        fc_baseline = tf.layers.Dense(1, name='fc_baseline')

        submodules = {'policyNet': policyNet,
                      'VAEencoder': VAEencoder,
                      'VAEdecoder': VAEdecoder}
        if FLAGS.planner == 'ActInf':
            planner = ActInfPlanner(FLAGS, submodules, self.B, env.patch_shape_flat, self.C, stateTransition_AC)
        elif FLAGS.planner == 'RL':
            planner = REINFORCEPlanner(FLAGS, submodules, self.B, env.patch_shape_flat)
        else:
            raise ValueError('Undefined planner.')

        self.n_policies = planner.n_policies

        # variables to remember. Probably to be implemented via TensorArray
        out_ta = []
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='glimpse_nlls_posterior'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='glimpse_reconstr'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='zs_post'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='G'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='actions'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='actions_mean'))
        out_ta.append(tf.TensorArray(tf.int32,   size=min_glimpses, dynamic_size=True, name='decisions'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='rewards'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='baselines'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses+1, dynamic_size=True, name='current_cs'))
        out_ta.append(tf.TensorArray(tf.bool,    size=min_glimpses, dynamic_size=True, name='done'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='exp_exp_obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='exp_obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='H_exp_exp_obs'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='exp_H'))
        out_ta.append(tf.TensorArray(tf.float32, size=min_glimpses, dynamic_size=True, name='potential_actions'))

        ta_d = {}
        for i, ta in enumerate(out_ta):
            ta_d[ta.handle.name.split('/')[-1].replace(':0', '')] = ta

        # Initial values
        last_done = tf.zeros([self.B], dtype=tf.bool)
        last_decision = tf.fill([self.B], -1)
        # in case starting calculation after initial observation (as first location should be identical for all images)
        next_action, next_action_mean = policyNet.inital_loc()
        next_decision = tf.fill([self.B], -1)
        current_state = stateTransition_AC.initial_state(self.B, next_action)

        ta_d['current_cs'] = write_zero_out(0, ta_d['current_cs'], current_state['c'], last_done)

        # out of loop to not create new tensors every step
        one_hot_label = tf.one_hot(tf.range(FLAGS.num_classes), depth=FLAGS.num_classes)
        one_hot_label_repeated = repeat_axis(one_hot_label, 0, self.B)  # [B * hyp, hyp]

        def current_belief_update(current_state, new_observation, exp_obs_prior, time):
            """Given a new observation, and the last believes over the state, update the believes over the states.
            The sufficient statistic of the old state in this case is z, as the VAEencoder is class-specific.

            Returns:
                c: [B, num_classes} believe over classes based on past observations
                zs_post: [B, num_classes, size_z] inferred zs conditional on each class
                glimpse_nll_stacked: [B, num_classes] likelihood of each past observation conditional on each class
                """
            with tf.name_scope('Belief_update'):
                # Infer posterior z for all hypotheses
                with tf.name_scope('poterior_inference_per_hyp'):
                    class_conditional_s = tf.reshape(current_state['s'], [self.B * FLAGS.num_classes, FLAGS.size_rnn])
                    new_action_repeated = repeat_axis(current_state['l'], 0, FLAGS.num_classes)
                    new_observation_repeated = repeat_axis(new_observation, 0, FLAGS.num_classes)

                    z_post = VAEencoder.posterior_inference(one_hot_label_repeated,
                                                            class_conditional_s,
                                                            tf.stop_gradient(new_action_repeated),
                                                            new_observation_repeated)
                    # 2 possibilties to infer state from received observations:
                    # i)  judge by likelihood of the observations under each hypothesis
                    # ii) train a separate model (e.g. LSTM) for infering states
                    # TODO: CAN WE DO THIS IN AN ENCODED SPACE?
                    posterior = VAEdecoder.decode(one_hot_label_repeated,
                                                  class_conditional_s,
                                                  z_post['sample'],
                                                  tf.stop_gradient(new_action_repeated),
                                                  new_observation_repeated)  # ^= filtering, given that transitions are deterministic

                    zs_post         = tf.reshape(tf.concat([z_post['mu'], z_post['sigma']], axis=1),
                                                 [self.B, FLAGS.num_classes, 2*FLAGS.size_z])
                    zs_post_samples = tf.reshape(z_post['sample'], [self.B, FLAGS.num_classes, FLAGS.size_z])
                    reconstr_post   = tf.reshape(posterior['sample'], [self.B, FLAGS.num_classes, env.patch_shape_flat])
                    nll_post        = tf.reshape(posterior['loss'], [self.B, FLAGS.num_classes])

                # believes over the classes based on all past observations (uniformly weighted)
                with tf.name_scope('belief_update'):
                    # TODO: THINK ABOUT THE SHAPE. PRIOR SHOULD BE FOR EACH HYP. USE new_observation_repeated?
                    prior_nll = calculate_gaussian_nll(exp_obs_prior, new_observation)

                    if time == 0:
                        c = tf.nn.softmax(-prior_nll, axis=1)
                    else:
                        c = (1. / time) * tf.nn.softmax(-prior_nll, axis=1) + (time - 1.) / time * current_state['c']

                return (c,  # [B, num_classes]
                        zs_post,  # [B, num_classes, 2*z]
                        zs_post_samples,  # [B, num_classes, z]
                        nll_post,  # [B, num_classes]
                        reconstr_post)  # [B, num_classes, glimpse]


        with tf.name_scope('Main_loop'):
            for time in range(FLAGS.num_glimpses):
                if time == 0:


                if time > 1:
                    if random_locations:
                        next_decision, next_action, next_action_mean, pl_records = planner.random_policy()
                    else:
                        next_decision, next_action, next_action_mean, next_exp_obs, pl_records = planner.planning_step(current_state, z_samples, time, self.is_training)

                    # TODO : Could REUSE FROM PLANNING STEP
                    current_state = stateTransition_AC([last_z, labels, next_action], last_state)

                observation, corr_classification_fb, done = env.step(next_action, next_decision)
                done = tf.logical_or(last_done, done)
                obs_enc = glimpseEncoder.encode(observation)


                current_state['c'], zs_post, z_samples, nll_posterior, reconstr_posterior = current_belief_update(current_state, obs_enc, next_exp_obs, time)
                # baseline = fc_baseline(tf.stop_gradient(tf.concat([current_c, tf.fill([self.B, 1], tf.cast(time, tf.float32))], axis=1)))
                baseline = tf.squeeze(fc_baseline(tf.stop_gradient(current_state['c'])), 1)

                # t=0 to T-1. ACTION RECORDING HAS TO STAY BEFORE PLANNING OR WILL BE OVERWRITTEN
                ta_d['obs']                      = write_zero_out(time, ta_d['obs'], observation, done)
                ta_d['zs_post']                  = write_zero_out(time, ta_d['zs_post'], zs_post, done)  # [B, n_policies, size_z]
                ta_d['glimpse_nlls_posterior']   = write_zero_out(time, ta_d['glimpse_nlls_posterior'], nll_posterior, done)  # [B, n_policies]
                ta_d['glimpse_reconstr']         = write_zero_out(time, ta_d['glimpse_reconstr'], reconstr_posterior, done)  # for visualisation only
                ta_d['actions']                  = write_zero_out(time, ta_d['actions'], next_action, done)  # location actions, not including the decision acions
                ta_d['actions_mean']             = write_zero_out(time, ta_d['actions_mean'], next_action_mean, done)  # location actions, not including the decision acions
                ta_d['baselines']                = write_zero_out(time, ta_d['baselines'], baseline, done)
                ta_d['done']                     = ta_d['done'].write(time, done)
                # t=0 to T
                ta_d['rewards']                  = write_zero_out(time, ta_d['rewards'] , corr_classification_fb, last_done)

                if random_locations:
                    next_decision, next_action, next_action_mean, pl_records = planner.random_policy()
                else:
                    next_decision, next_action, next_action_mean, pl_records = planner.planning_step(current_state, zs_post, z_samples, time, self.is_training)

                # t=1 to T
                for k, v in pl_records.items():
                    ta_d[k] = write_zero_out(time, ta_d[k], v, last_done)
                ta_d['current_cs'] = write_zero_out(time+1, ta_d['current_cs'], current_state['c'], last_done)  # ONLY ONE t=0 TO T
                ta_d['decisions']  = write_zero_out(time, ta_d['decisions'], next_decision, last_done)
                # copy forward
                classification_decision = tf.where(last_done, last_decision, next_decision)
                # pass on to next time step
                last_done = done
                last_decision = next_decision
                last_z = zs_post  # TODO: or should this be the sampled ones?
                # last_c = current_c  # TODO: could also use the one from planning (new_c) or pi
                # last_s = current_s

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
            self.glimpse_nlls_posterior = ta_d['glimpse_nlls_posterior'].stack()  # [T,B,hyp]
            zs_post = ta_d['zs_post'].stack()  # [T,B,hyp,2*z]
            self.state_believes = ta_d['current_cs'].stack()  # [T+1,B,hyp]
            self.G = ta_d['G'].stack()  # not zero'd-out so far!
            bl_loc = ta_d['baselines'].stack()
            self.glimpse_reconstr = ta_d['glimpse_reconstr'].stack()  # [T,B,hyp,glimpse]

            # further records for debugging
            self.exp_exp_obs = ta_d['exp_exp_obs'].stack()
            self.exp_obs = ta_d['exp_obs'].stack()
            self.H_exp_exp_obs = ta_d['H_exp_exp_obs'].stack()
            self.exp_H = ta_d['exp_H'].stack()
            self.potential_actions = ta_d['potential_actions'].stack()  # [T,B,n_policies,loc]

            self.num_glimpses_dyn = tf.shape(self.obs)[0]
            T = FLAGS.num_glimpses - tf.count_nonzero(done, 0, dtype=tf.float32)
            self.avg_T = tf.reduce_mean(T)

        with tf.name_scope('Losses'):
            with tf.name_scope('RL'):
                returns = tf.cumsum(rewards, reverse=True, axis=0)
                policy_losses = policyNet.REINFORCE_losses(returns, bl_loc, self.actions, actions_mean)  # [T,B]
                policy_loss   = tf.reduce_sum(tf.reduce_mean(policy_losses, 1))

                baseline_mse = tf.reduce_mean(tf.square(tf.stop_gradient(returns[1:]) - bl_loc[:-1]))

            with tf.name_scope('Classification'):
                # might never make a classification decision
                # TODO: SHOULD I FORCE THE ACTION AT t=t TO BE A CLASSIFICATION?
                self.classification = classification_decision

            with tf.name_scope('VAE'):
                # mask losses of wrong hyptheses
                nll_posterior = tf.reduce_sum(self.glimpse_nlls_posterior, 0)  # sum over time
                correct_hypoths = tf.cast(tf.one_hot(env.y_MC, depth=FLAGS.num_classes), tf.bool)
                nll_posterior = tf.where(correct_hypoths, nll_posterior, tf.zeros_like(nll_posterior))  # zero-out all but true hypothesis
                nll_posterior = tf.reduce_mean(nll_posterior)  # mean over batch

                # assume N(0,1) prior model (event though atm prior never used)
                prior_mu = tf.fill([self.B, FLAGS.size_z], 0.)
                prior_sigma = tf.fill([self.B, FLAGS.size_z], 1.)

                zs_post_correct = tf.boolean_mask(zs_post, correct_hypoths, axis=1)
                post_mu, post_sigma = tf.split(zs_post_correct, 2, axis=2)
                # KL_div = T * VAEencoder.kl_div_normal(post_mu, post_sigma, prior_mu, prior_sigma)  # NOTE: "T *" is wrong as T is [self.B]. Incorporat before reducing to a scalar
                N_post = tfd.Normal(loc=post_mu, scale=post_sigma)
                N_prior = tfd.Normal(loc=prior_mu, scale=prior_sigma)
                KL_div = N_post.kl_divergence(N_prior)
                KL_div = tf.where(tf.tile(done[:, :, tf.newaxis], [1, 1, FLAGS.size_z]), tf.zeros_like(KL_div), KL_div)  # replace those that are done
                KL_div = tf.reduce_mean(tf.reduce_sum(KL_div, 0))

            # TODO: SCALE LOSSES DIFFERENTLY? (only necessary if they flow into the same weights, might not be the case so far)
            self.loss = policy_loss + baseline_mse + nll_posterior + KL_div


        with tf.variable_scope('Optimizer'):
            if random_locations:
                pretrain_vars = VAEencoder.trainable + VAEdecoder.trainable
                self.train_op, gradient_check_Pre, _ = self._create_train_op(FLAGS, nll_posterior + KL_div, self.global_step, varlist=pretrain_vars)
            else:
                self.train_op, gradient_check_F, _ = self._create_train_op(FLAGS, self.loss, self.global_step)

        with tf.name_scope('Summaries'):
            metrics_upd_coll = "streaming_updates"

            scalars = {'loss/loss': self.loss,
                       'loss/accuracy': tf.reduce_mean(tf.cast(tf.equal(classification_decision, self.y_MC), tf.float32)),
                       'loss/VAE_nll_posterior': nll_posterior,
                       'loss/VAE_KL_div': KL_div,
                       'loss/RL_loc_baseline_mse': tf.reduce_mean(baseline_mse),
                       'loss/RL_policy_loss': policy_loss,
                       'loss/RL_returns': tf.reduce_mean(returns),
                       'misc/T': self.avg_T,
                       'misc/share_no_decision': tf.count_nonzero(tf.equal(classification_decision, -1), dtype=tf.float32) / tf.cast(self.B, tf.float32)}

            for name, scalar in scalars.items():
                tf.summary.scalar(name, scalar)
                tf.metrics.mean(scalar, name=name, updates_collections=metrics_upd_coll)

            self.metrics_update = tf.get_collection(metrics_upd_coll)
            self.metrics_names = [v.name.replace('_1/update_op:0', '').replace('Summaries/', '') for v in self.metrics_update]

            self.summary = tf.summary.merge_all()

            self.glimpses_composed = env.composed_glimpse(FLAGS, self.obs, self.num_glimpses_dyn)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(classification_decision, self.y_MC), tf.float32))  # only to get easy direct intermendiate outputs

        self.saver = self._create_saver(phase)

        ##################################
        ### SKETCH OF VISUAL_FORAGING CODE
        ##################################
        # def planning_sketch(last_l, c, next_actions):
        #     for k, action in enumerate(next_actions):
        #         # action specific state transition
        #         new_c, new_z, new_l = self.state_transition(c, zs_post[time, :, k], last_l, action)
        #
        #         G = 0
        #         overall_exp_obs = tf.zeros(self.glimpse_size)
        #         for hyp in range(FLAGS.num_classes):  # could save computation by only doing this for non-negligible hypothesis
        #             exp_obs = VAEdecoder(new_l, hyp, new_z[hyp])
        #
        #             overall_exp_obs += new_c[hyp] * exp_obs
        #             # check if interpreting correctly or transpose is needed( or even easier, use log probabilities and sigmoid entropy
        #             # entropy over pixel space might not make that much sense
        #             G += new_c[hyp] * exp_obs * tf.log(exp_obs)  # should return a scalar
        #
        #         G -= overall_exp_obs * tf.log(overall_exp_obs)
        #
        #         # incorporate prior preferences over outcomes
        #         G += overall_exp_obs * C
