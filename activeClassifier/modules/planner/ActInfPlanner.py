import tensorflow as tf
from tensorflow import distributions as tfd

from activeClassifier.modules.planner.basePlanner import BasePlanner
from activeClassifier.tools.tf_tools import repeat_axis, binary_entropy, differential_entropy_normal, differential_entropy_diag_normal


class ActInfPlanner(BasePlanner):
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition, C):
        super().__init__(FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition)
        self.n_policies = 1 if FLAGS.rl_reward == 'G1' else FLAGS.num_classes_kn
        self.num_classes_kn = FLAGS.num_classes_kn
        self.FE_dist = 'B' if FLAGS.use_pixel_obs_FE else FLAGS.z_dist
        self.use_pixel_obs_FE = FLAGS.use_pixel_obs_FE
        self.pixel_obs_discrete = FLAGS.pixel_obs_discrete
        self.exp_obs_shape = self.m['VAEDecoder'].output_shape if self.use_pixel_obs_FE else self.m['VAEEncoder'].output_shape_flat
        self.B = batch_sz
        self.C = C
        self.alpha = FLAGS.precision_alpha
        self.actInfPolicy = FLAGS.actInfPolicy

        # No need to tile at every step
        self.hyp = self._hyp_tiling(FLAGS.num_classes_kn, self.B * self.n_policies)  # [B * n_policies * hyp, num_classes]
        self.policy_dep_input = self._hyp_tiling(self.num_classes_kn, self.B)  # [B * hyp, num_classes]

    @staticmethod
    def _discrete_entropy_agg(d, logits=None, probs=None, agg=True):
        # TODO: DOES MEAN MAKE SENSE? (at least better than sum, as indifferent to size_z)
        if d == 'B':
            dist = tfd.Bernoulli(logits=logits, probs=probs)
        elif d == 'Cat':
            dist = tfd.Categorical(logits=logits, probs=probs)
        H = dist.entropy()
        if agg:
            H = tf.reduce_mean(H, axis=-1)  # [B, n_policies, hyp]
        return H

    @staticmethod
    def _believe_weighted_expecation(probs, believes=None, believes_tiled=None):
        if (believes is None) == (believes_tiled is None):
            raise ValueError('Provide either c_believes or c_believes_tiled')

        if believes is not None:
            exp_probs = tf.einsum('bh,bkhg->bkg', believes, probs)  # [B, n_policies, glimpse]
        else:
            exp_probs = tf.einsum('bkh,bkhg->bkg', believes_tiled, probs)  # [B, n_policies, glimpse]
        return exp_probs

    def calc_G_obs_prePreferences(self, exp_obs_logits, exp_obs_sigma, c_believes=None, c_believes_tiled=None):
        """
        Args:
            exp_obs: dict with variable of shape [B, n_policies, hyp, glimpse]
            c_believes: [B, hyp]
            c_believes_tiled: [B, n_policies, hyp]
        Returns:
            G, exp_exp_obs, exp_H, H_exp_exp_obs: [B, n_policies]
        """
        # expected entropy
        if (c_believes is None) == (c_believes_tiled is None):
            raise ValueError('Provide either c_believes or c_believes_tiled')

        if self.pixel_obs_discrete:
            H = self._discrete_entropy_agg(logits=exp_obs_logits, d='Cat')  # [B, n_policies, hyp]
            exp_obs = tf.nn.softmax(exp_obs_logits)  # Cannot take the expectation on the logits due to nonlinearity of sigmoid!
            exp_obs = tf.reshape(exp_obs, [self.B, self.n_policies, self.num_classes_kn, -1])
            exp_exp_obs = self._believe_weighted_expecation(exp_obs, c_believes, c_believes_tiled)  # [B, n_policies, glimpse]
            exp_exp_obs = tf.reshape(exp_exp_obs, [self.B, self.n_policies] + self.exp_obs_shape)
            H_exp_exp_obs = self._discrete_entropy_agg(probs=exp_exp_obs, d='Cat')  # [B, n_policies]
        elif self.FE_dist == 'B':
            H = self._discrete_entropy_agg(logits=exp_obs_logits, d='B')  # [B, n_policies, hyp]
            exp_obs = tf.nn.sigmoid(exp_obs_logits)  # Cannot take the expectation on the logits due to nonlinearity of sigmoid!
            exp_exp_obs = self._believe_weighted_expecation(exp_obs, c_believes, c_believes_tiled)  # [B, n_policies, glimpse]
            H_exp_exp_obs = self._discrete_entropy_agg(probs=exp_exp_obs, d='B')  # [B, n_policies]
        elif self.FE_dist == 'N':
            H = differential_entropy_diag_normal(exp_obs_sigma)  # [B, n_policies, hyp]
            raise ValueError('Not fully implemented yet')

        if c_believes is not None:
            c_believes_tiled = c_believes[:, tf.newaxis, :]  # [B, hyp] -> [B, 1, hyp] -> broadcasting into [B, n_policies, hyp]
        exp_H = tf.reduce_sum(c_believes_tiled * H, axis=2)  # [B, n_policies]

        G = H_exp_exp_obs - exp_H
        return G, exp_exp_obs, exp_H, H_exp_exp_obs

    def _G_decision(self, time, c_believes):
        # decision actions: will always assign 0 probabiity to any of the non-decsion actions, i.e. those won't impact the entropy
        # TODO: SHOULD THIS DIFFER FOR MULTI-STEP PLANNING? E.G. SHOULD THE STATE BELIEVES INCORPORATE THE FORECASTED OBSERVATIONS?
        # TODO: SHOULD PROBABLY TAKE INTO ACCOUNT THE UK BELIEF
        believe_correct = tf.reduce_max(c_believes, axis=1)
        dec_extr_value = believe_correct * self.C[time, 1] + (1. - believe_correct) * self.C[time, 2]
        dec_H = binary_entropy(probs=believe_correct)  # probs will be zero for anything but the feedback outcomes, i.e. irrelevant for entropy
        dec_G = dec_extr_value - dec_H
        return dec_G

    def _action_selection(self, next_actions, next_actions_mean, new_state, G, exp_obs_prior, time, is_training):
        # TODO: should uniform_loc10 take random decisions or not?
        if self.actInfPolicy in ['random', 'uniform_loc10']:
            selected_action_idx = tf.random_uniform(shape=[self.B], minval=0, maxval=self.n_policies, dtype=tf.int32)
            if time < (self.num_glimpses - 1):
                decision = tf.fill([self.B], -1)
            else:
                decision = self._best_believe(new_state)
        else:
            # TODO: use pi = softmax(-F - gamma*G) instead?
            gamma = 1
            pi_logits = tf.nn.log_softmax(gamma * G)
            # Incorporate past into the decision. But what to use for the 2 decision actions?
            # pi_logits = tf.nn.log_softmax(tf.log(new_state['c']) + gamma * G)

            # TODO: precision?
            # TODO: which version of action selection?
            # Visual foraging code: a_t ~ softmax(alpha * log(softmax(-F - gamma * G))) with alpha=512
            # Visual foraging paper: a_t = min_a[ o*_t+1 * [log(o*_t+1) - log(o^a_t+1)] ]
            # Maze code: a_t ~ softmax(gamma * G) [summed over policies with the same next action]
            selected_action_idx = tf.cond(is_training,
                                          lambda: tfd.Categorical(logits=self.alpha * pi_logits, allow_nan_stats=False).sample(),
                                          lambda: tf.argmax(G, axis=1, output_type=tf.int32),
                                          name='sample_action_cond')
            # give back the action itself, not its index. Differentiate between decision and location actions
            best_belief = self._best_believe(new_state)
            dec = tf.equal(selected_action_idx, self.n_policies)  # the last action is the decision
            selected_action_idx = tf.where(tf.stop_gradient(dec), tf.fill([self.B], 0), selected_action_idx)  # replace decision indeces (which exceed the shape of selected_action), so we can use gather on the locations

            decision = tf.cond(tf.equal(time, self.num_glimpses - 1),
                               lambda: best_belief,  # always take a decision at the last time step
                               lambda: tf.where(dec, best_belief, tf.fill([self.B], -1)),
                               name='last_t_decision_cond')

        if self.n_policies == 1:
            selected_action, selected_action_mean = next_actions, next_actions_mean
            selected_exp_obs = {k: tf.reshape(v, [self.B, self.num_classes_kn, v.shape[-1]]) if (v is not None) else None
                                for k, v in exp_obs_prior.items()}  # squeeze out policy dim (squeeze would turn shape into unknown)
        else:
            coords = tf.stack(tf.meshgrid(tf.range(self.B)) + [selected_action_idx], axis=1)
            selected_action      = tf.gather_nd(next_actions, coords)
            selected_action_mean = tf.gather_nd(next_actions_mean, coords)
            selected_exp_obs     = {k: tf.gather_nd(v, coords) if (v is not None) else None for k, v in exp_obs_prior.items()}  # [B, num_classes_kn, -1] as n_policies get removed in gather_nd
        return decision, selected_action, selected_action_mean, selected_exp_obs, selected_action_idx

    def _location_planning(self, inputs, is_training, rnd_loc_eval):
        if self.actInfPolicy == 'uniform_loc10':
            next_actions, next_actions_mean = self.m['policyNet'].uniform_loc_10(self.n_policies)
        elif self.actInfPolicy == "random":
            next_actions, next_actions_mean = self.m['policyNet'].random_loc(shp=[self.B, self.n_policies])
        else:
            next_actions, next_actions_mean = self.m['policyNet'].next_actions(inputs=inputs, is_training=is_training,
                                                                               n_policies=self.n_policies)  # [B, n_policies, loc_dim]

            next_actions, next_actions_mean = tf.cond(rnd_loc_eval,
                                                      # lambda: self.m['policyNet'].random_loc(shp=[self.B, self.n_policies] if (self.n_policies > 1) else [self.B]),
                                                      lambda: self.m['policyNet'].uniform_loc_10(self.n_policies) if (self.n_policies > 1) else self.m['policyNet'].random_loc(shp=[self.B]),
                                                      lambda: (next_actions, next_actions_mean),
                                                      name='rnd_loc_eval_cond')
        return next_actions, next_actions_mean

    def planning_step(self, current_state, time, is_training, rnd_loc_eval):
        """Perform one planning step.
        Args:
            current state

        Returns:
            Next state
        """
        with tf.name_scope('Planning_loop/'):  # loop over policies, parallised into [B * self.n_policies, ...]
            # TODO: define inputs for policyNet (and use the same in reinforce-planner if using it for pre-training)`
            # inputs = [current_state['s'], tf.fill([self.B, 1], tf.cast(time, tf.float32))]
            if self.n_policies == 1:  # 'G1'
                inputs = [current_state['s'], current_state['c']]
            elif self.rl_reward == 'clf':
                assert self.n_policies == self.num_classes_kn
                inputs = [repeat_axis(current_state['s'], axis=0, repeats=self.num_classes_kn), self.policy_dep_input]
            elif self.rl_reward == 'G':
                assert self.n_policies == self.num_classes_kn
                boosted_hyp = repeat_axis(current_state['c'], axis=0, repeats=self.n_policies)  # [B * n_policies, num_classes]
                # increase each 'hypothesis'-policy by 50% and renormalise
                boosted_hyp += self.policy_dep_input * boosted_hyp * 0.5
                def re_normalise(x, axis=-1):
                    return x / tf.reduce_sum(x, axis=axis, keep_dims=True)
                boosted_hyp = re_normalise(boosted_hyp, axis=-1)
                inputs = [repeat_axis(current_state['s'], axis=0, repeats=self.n_policies), boosted_hyp]
            else:
                raise ValueError('Unknown policy strategies', 'n_policies: {}, rl_reward: {}'.format(self.n_policies, self.rl_reward))

            # select locations to evaluate
            next_actions, next_actions_mean = self._location_planning(inputs, is_training, rnd_loc_eval)

            with tf.name_scope('Hypotheses_loop/'):  # for every action: loop over hypotheses
                s_tiled = repeat_axis(current_state['s'], axis=0, repeats=self.n_policies * self.num_classes_kn)  # [B, rnn] -> [B * n_policies * hyp, rnn]
                # TODO: THIS MIGHT HAVE TO CHANGE IF POLICIES DEPEND ON CLASS-CONDITIONAL Z
                next_actions_tiled = repeat_axis(tf.reshape(next_actions, [self.B * self.n_policies, self.loc_dim]),
                                          axis=0, repeats=self.num_classes_kn)  # [B, n_policies, loc] -> [B * n_policies, loc] -> [B * n_policies * hyp, loc]
                exp_obs_prior_enc = self.m['VAEEncoder'].calc_prior([self.hyp, s_tiled, next_actions_tiled],
                                                                    out_shp=[self.B, self.n_policies, self.num_classes_kn])

                if not self.use_pixel_obs_FE:
                    exp_obs_prior_logits = exp_obs_prior_enc['mu']
                    exp_obs_prior_sigma = exp_obs_prior_enc['sigma']
                    sample = exp_obs_prior_enc['sample']
                else:
                    exp_obs_prior = self.m['VAEDecoder'].decode([tf.reshape(exp_obs_prior_enc['sample'], [-1] + self.m['VAEEncoder'].output_shape_flat), next_actions_tiled],
                                                                out_shp=[self.B, self.n_policies, self.num_classes_kn])
                    exp_obs_prior_logits = exp_obs_prior['mu_logits']
                    exp_obs_prior_sigma = exp_obs_prior['sigma']
                    sample = exp_obs_prior['sample']

            G_obs, exp_exp_obs, exp_H, H_exp_exp_obs = self.calc_G_obs_prePreferences(exp_obs_prior_logits, exp_obs_prior_sigma, c_believes=current_state['c'])
            # For all non-decision actions the probability of classifying is 0, hence the probability of an observation is 1
            preference_error_obs = 1. * self.C[time, 0]
            G_obs += preference_error_obs
            # decision actions
            G_dec = self._G_decision(time, current_state['c'])
            G = tf.concat([G_obs, G_dec[:, tf.newaxis]], axis=1)

            # action selection
            decision, selected_action, selected_action_mean, selected_exp_obs_enc, selected_action_idx = self._action_selection(next_actions, next_actions_mean, current_state, G, exp_obs_prior_enc, time, is_training)

            if self.rl_reward == 'G':
                boosted_hyp = tf.reshape(boosted_hyp, [self.B, self.n_policies, self.num_classes_kn])
                rewards_Gobs, _, rewards_exp_H, rewards_H_exp_exp_obs = self.calc_G_obs_prePreferences(exp_obs_prior_logits, exp_obs_prior_sigma, c_believes_tiled=boosted_hyp)
                r = rewards_Gobs
            else:
                r = tf.zeros([self.B, self.n_policies])

        records = {'G'                : G,
                   'exp_obs'          : sample,  # [B, n_policies, num_classes, z]
                   'exp_exp_obs'      : exp_exp_obs,  # [B, n_policies, z]
                   'H_exp_exp_obs'    : H_exp_exp_obs,
                   'exp_H'            : exp_H,
                   'potential_actions': next_actions[:, tf.newaxis, :] if (self.n_policies == 1) else next_actions,
                   'selected_action_idx': selected_action_idx,
                   'rewards_Gobs'     : r}
        return decision, selected_action, selected_action_mean, selected_exp_obs_enc, records
