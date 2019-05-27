import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from visualisation.base import Base, visualisation_level

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Visualization_predRSSM(Base):
    def __init__(self, model, FLAGS):
        super().__init__(model, FLAGS)

        self.fetch = {'step'            : model.global_step,
                      'epoch'           : model.epoch_num,
                      'phase'           : model.phase,
                      'x'               : model.x_MC,
                      'y'               : model.y_MC,
                      'locs'            : model.actions,
                      'gl_composed'     : model.glimpses_composed,
                      'clf'             : model.classification,
                      'decisions'       : model.decisions,
                      'state_believes'  : model.state_believes,
                      'G'               : model.G,
                      'nll_posterior'   : model.nll_posterior,
                      'glimpse'         : model.obs,
                      'reconstr_posterior': model.reconstr_posterior,
                      'reconstr_prior'  : model.reconstr_prior,
                      'KLdivs'          : model.KLdivs,
                      'fb'              : model.fb,
                      'uk_belief'       : model.uk_belief,
                      'potential_actions': model.potential_actions,
                      'H_exp_exp_obs'   : model.H_exp_exp_obs,
                      'exp_H'           : model.exp_H,
                      'exp_exp_obs'     : model.exp_exp_obs,
                      'selected_exp_obs': model.selected_exp_obs,
                      'z_post'          : model.z_post,
                      'selected_action_idx': model.selected_action_idx,
                      }

    @visualisation_level(1)
    def visualise(self, sess, feed, suffix='', nr_obs_overview=8, nr_obs_reconstr=5):
        d = self._eval_feed(sess, feed)

        nr_obs_overview = min(nr_obs_overview, self.batch_size_eff)  # batch_size_eff is set in _eval_feed() -> has to come before
        nr_obs_reconstr = min(nr_obs_reconstr, self.batch_size_eff)

        self.plot_overview(d, nr_obs_overview, suffix)

        self.plot_stateBelieves(d, suffix)
        self.plot_reconstr(d, nr_obs_reconstr, suffix)

    #     self.plot_fb(d, prefix)
        if (self.planner == 'ActInf') & (d['epoch'] >= self.pre_train_epochs):
            # self.plot_planning(d, nr_examples=nr_obs_reconstr)
            self.plot_planning_patches(d, nr_examples=nr_obs_reconstr)
            self.plot_z(d, 3, suffix)

    @visualisation_level(1)
    def plot_reconstr(self, d, nr_examples, suffix='', folder_name='reconstr'):
        def get_title_color(post_believes, hyp):
            if post_believes[hyp] == post_believes.max():
                color = 'magenta'
            elif post_believes[hyp] > 0.1:
                color = 'blue'
            else:
                color = 'black'
            return color

        nax = 2 + self.num_classes_kn

        gl = self._scale_reshp(d['glimpse'])  # [T, B, scale[0], scales*scale[0]]
        gl_post = self._scale_reshp(d['reconstr_posterior'])  # [T, B, scale[0], scales*scale[0]]
        gl_preds = self._scale_reshp(d['reconstr_prior'])  # [T, B, hyp, scale[0], scales*scale[0]]

        idx_examples = self._get_idx_examples(d['y'], nr_examples, replace=False)

        for i in idx_examples:
            f, axes = plt.subplots(self.num_glimpses + 1, nax, figsize=(4 * self.num_scales * nax, 4 * (self.num_glimpses + 1)))
            axes    = axes.reshape([self.num_glimpses + 1, nax])
            self._plot_img_plus_locs(axes[0, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])

            for t in range(self.num_glimpses):
                # true glimpse
                axes[t+1, 0].imshow(gl[t, i], cmap='gray')
                title = 'Label: {}, clf: {}'.format(self.lbl_map[d['y'][i]], self.lbl_map[d['clf'][i]])
                if self.uk_label is not None:
                    title += ', p(uk): {:.2f}'.format(d['uk_belief'][t, i])
                axes[t+1, 0].set_title(title)
                # posterior
                axes[t+1, 1].imshow(gl_post[t, i], cmap='gray')
                axes[t+1, 1].set_title('Posterior, nll: {:.2f}'.format(d['nll_posterior'][t, i]))
                # prior for all classes
                ranked_losses = np.argsort(d['KLdivs'][t, i, :])
                ps = softmax(-d['KLdivs'][t, i, :])
                for j, hyp in enumerate(ranked_losses):
                    axes[t+1, j+2].imshow(gl_preds[t, i, hyp], cmap='gray')
                    if d['decisions'][t, i] != -1:
                        axes[t+1, j + 2].set_title('Decision: {}'.format(d['decisions'][t, i]))
                    else:
                        c = get_title_color(d['state_believes'][t+1, i, :], hyp)
                        axes[t+1, j + 2].set_title('{}, p: {:.2f}, KL: {:.2f}, post-c: {:.2f}'.format(self.lbl_map[hyp], ps[hyp], d['KLdivs'][t, i, hyp], d['state_believes'][t+1, i, hyp]), color=c)

            [ax.set_axis_off() for ax in axes.ravel()]
            self._save_fig(f, folder_name, '{}{}_n{}{isuk}.png'.format(self.prefix, suffix, i,
                                                                       isuk='_uk' if (d['y'][i] == self.uk_label) else ''))

    # @visualisation_level(2)
    # def plot_planning(self, d, nr_examples, suffix='', folder_name='planning'):
    #     # T x [True glimpse, exp_exp_obs, exp_obs...]
    #     nax_x = self.num_policies
    #     nax_y = 1 + self.num_glimpses
    #
    #     # exp_exp_obs = self._scale_reshp(d['exp_exp_obs'])  # [T, B, n_policies, scale[0], scales*scale[0]]
    #     # exp_obs = self._scale_reshp(d['exp_obs'])  # [T, B, n_policies, num_classes, scale[0], scales*scale[0]]
    #
    #     for i in range(nr_examples):
    #         f, axes = plt.subplots(nax_y, nax_x, figsize=(4 * self.num_scales * nax_x, 4 * nax_y))
    #         axes = axes.reshape([nax_y, nax_x])
    #         self._plot_img_plus_locs(axes[0, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])
    #
    #         # Note: first action is random, meaning d['potential_actions'][0] will be zero
    #         for t in range(self.num_glimpses):
    #             for k in range(self.num_policies):
    #                 # potential location under evaluation
    #                 locs = d['potential_actions'][t, i, k]
    #                 color = 'green' if (locs == d['locs'][t, i, :]).all() else 'cyan'
    #
    #                 axes[t, k].imshow(d['x'][i].reshape(self.img_shape_squeezed), cmap='gray')
    #                 axes[t, k].scatter(locs[1], locs[0], marker='x', facecolors=color, linewidth=2.5, s=0.25 * (5 * 8 * 24))
    #                 axes[t, k].add_patch(Rectangle(locs[::-1] - self.scale_sizes[0] / 2, width=self.scale_sizes[0], height=self.scale_sizes[0], edgecolor=color, facecolor='none', linewidth=2.5))
    #                 axes[t, k].set_title('G: {:.2f}, H_: {:.2f}, exp_H: {:.2f}, G_dec: {:.2f}'.format(d['G'][t, i, k], d['H_exp_exp_obs'][t, i, k], d['exp_H'][t, i, k], d['G'][t, i, -1]))
    #
    #                 # ranked_hyp = np.argsort(d['state_believes'][t, i, :])
    #                 # for j, hyp in enumerate(ranked_hyp[::-1]):
    #                 #     # axes[t, j + 2].imshow(exp_obs[t, i, k, hyp], cmap='gray')
    #                 #     axes[t, j + 2].set_title('Hyp: {}, prob: {:.2f}'.format(hyp, d['state_believes'][t, i, hyp]))
    #
    #         [ax.set_axis_off() for ax in axes.ravel()]
    #         self._save_fig(f, folder_name, '{}{}_n{}.png'.format(self.prefix, suffix, i))

    @visualisation_level(2)
    def plot_planning(self, d, nr_examples, suffix='', folder_name='planning'):
        # T x [True glimpse, exp_exp_obs, exp_obs...]
        nax_x = nr_examples
        nax_y = self.num_glimpses

        f, axes = plt.subplots(nax_y, nax_x, figsize=(8 * self.num_scales * nax_x, 4 * nax_y), squeeze=False)
        for i in range(nr_examples):
            # Note: first action is random, meaning d['potential_actions'][0] will be zero
            for t in range(self.num_glimpses):
                if t == 0:  # random action
                    self._plot_img_plus_locs(axes[0, i], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])
                    axes[t, i].set_title('t: {}, random policy, lbl: {}, clf: {}'.format(t, d['y'][i], d['clf'][i]))
                else:
                    if np.sum(d['H_exp_exp_obs'][t, i, :]) == 0.:
                        axes[t, i].set_title('t: {}, decision - no glimpse'.format(t))
                        break

                    axes[t, i].imshow(d['x'][i].reshape(self.img_shape_squeezed), cmap='gray')
                    axes[t, i].set_title('t: {}, selected policy: {}'.format(t, np.argmax(d['G'][t, i, :])))
                    for k in range(self.num_policies):
                        # potential location under evaluation
                        locs = d['potential_actions'][t, i, k]
                        color = 'C{}'.format(k)
                        correct = np.all((locs == d['locs'][t, i, :]))

                        lbl = '{}: G: {:.2f}, H_: {:.2f}, exp_H: {:.2f}, G_dec: {:.2f}'.format(k, d['G'][t, i, k], d['H_exp_exp_obs'][t, i, k], d['exp_H'][t, i, k], d['G'][t, i, -1])
                        axes[t, i].add_patch(Rectangle(locs[::-1] - self.scale_sizes[0] / 2,
                                                           width=self.scale_sizes[0], height=self.scale_sizes[0],
                                                           edgecolor=color, facecolor='none', linewidth=1.5, label=lbl))
                        if correct:
                            axes[t, i].scatter(locs[1], locs[0], marker='x', facecolors=color, linewidth=1.5, s=0.25 * (5 * 8 * 24))
                    # add current believes to legend
                    ranked_believes = np.argsort(- d['state_believes'][t, i, :])
                    lbl = 'hyp: ' + ', '.join('{} ({:.2f})'.format(j,  d['state_believes'][t, i, j]) for j in ranked_believes[:5])
                    axes[t, i].scatter(0, 0, marker='x', facecolors='k', linewidth=0, s=0, label=lbl)

                    chartBox = axes[t, i].get_position()
                    axes[t, i].set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
                    axes[t, i].legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)

        [ax.set_axis_off() for ax in axes.ravel()]
        self._save_fig(f, folder_name, '{}{}.png'.format(self.prefix, suffix))

    @visualisation_level(2)
    def plot_planning_patches(self, d, nr_examples, suffix='', folder_name='planning_patches'):
        nax_x = nr_examples
        nax_y = self.num_glimpses

        f, axes = plt.subplots(nax_y, nax_x, figsize=(8 * self.num_scales * nax_x, 4 * nax_y), squeeze=False)
        for i in range(nr_examples):
            # Note: first action is random, meaning d['potential_actions'][0] will be zero
            for t in range(self.num_glimpses):
                if t == 0:  # random action
                    self._plot_img_plus_locs(axes[0, i], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])
                    axes[t, i].set_title('t: {}, random policy, lbl: {}, clf: {}'.format(t, d['y'][i], d['clf'][i]))
                else:
                    # plot patches seen until now
                    self._plot_seen(d['x'][i], d['locs'][:, i], until_t=t, ax=axes[t, i])

                    # add current believes to legend
                    ranked_believes = np.argsort(- d['state_believes'][t, i, :])
                    lbl = 'hyp: ' + ', '.join('{} ({:.2f})'.format(j,  d['state_believes'][t, i, j]) for j in ranked_believes[:5])
                    axes[t, i].scatter(0, 0, marker='x', linewidth=0, s=0, label=lbl)

                    if np.sum(d['H_exp_exp_obs'][t, i, :]) == 0.:
                        axes[t, i].set_title('t: {}, decision - no new glimpse'.format(t))
                    else:
                        axes[t, i].set_title('t: {}, selected policy: {}'.format(t, np.argmax(d['G'][t, i, :])))

                        # plot rectangles for evaluated next locations
                        for k in range(self.num_policies):
                            # potential location under evaluation
                            locs = d['potential_actions'][t, i, k]
                            color = 'C{}'.format(k)
                            correct = np.all((locs == d['locs'][t, i, :]))

                            lbl = '{}: G: {:.2f}, H(exp): {:.2f}, E(H): {:.2f}, G_dec: {:.2f}'.format(k, d['G'][t, i, k], d['H_exp_exp_obs'][t, i, k], d['exp_H'][t, i, k], d['G'][t, i, -1])
                            axes[t, i].add_patch(Rectangle(locs[::-1] - self.scale_sizes[0] / 2,
                                                               width=self.scale_sizes[0], height=self.scale_sizes[0],
                                                               edgecolor=color, facecolor='none', linewidth=1.5, label=lbl))
                            if correct:
                                axes[t, i].scatter(locs[1], locs[0], marker='x', facecolors=color, linewidth=1.5, s=0.25 * (5 * 8 * 24))

                    # place legend next to plot
                    chartBox = axes[t, i].get_position()
                    axes[t, i].set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
                    axes[t, i].legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)

                    if np.sum(d['H_exp_exp_obs'][t, i, :]) == 0.:
                        break

        [(ax.set_xticks([]), ax.set_yticks([]), ax.set_ylim([self.img_shape[0] - 1, 0]), ax.set_xlim([0, self.img_shape[1] - 1]))  for ax in axes.ravel()]
        self._save_fig(f, folder_name, '{}{}.png'.format(self.prefix, suffix))

    @visualisation_level(2)
    def plot_z(self, d, nr_examples, suffix='', folder_name='z'):
        # T x [True glimpse, posterior, exp_exp_obs, exp_obs...]
        nax_x = 3 + self.num_classes_kn
        nax_y = self.num_glimpses

        gl = self._scale_reshp(d['glimpse'])  # [T, B, scale[0], scales*scale[0]]
        if self.size_z == 10:
            shp = [5, 2]
        elif self.size_z == 32:
            shp = [8, 4]
        elif self.size_z == 128:
            shp = [16, 8]
        else:
            shp = 2 * [int(np.sqrt(self.size_z))]
            if np.prod(shp) != self.size_z:
                print('Unspecified shape for this size_z and plot_z. Skipping z plots.')
                return
        z_post =  np.reshape(d['z_post'], [self.num_glimpses, self.batch_size_eff] + shp)
        exp_exp_obs = np.reshape(d['exp_exp_obs'], [self.num_glimpses, self.batch_size_eff, self.num_policies] + shp)
        exp_obs_prior = np.reshape(d['selected_exp_obs'], [self.num_glimpses, self.batch_size_eff, self.num_classes_kn] + shp)

        for i in range(nr_examples):
            f, axes = plt.subplots(nax_y, nax_x, figsize=(4 * self.num_scales * nax_x, 4 * nax_y), squeeze=False)
            for t in range(self.num_glimpses):
                if t == 0:
                    self._plot_img_plus_locs(axes[t, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])
                else:
                    axes[t, 0].imshow(gl[t, i], cmap='gray')
                    axes[t, 0].set_title('t: {}'.format(t))

                    axes[t, 1].imshow(z_post[t, i], cmap='gray')
                    axes[t, 1].set_title('z_post')

                    p = d['selected_action_idx'][t, i]
                    axes[t, 2].imshow(exp_exp_obs[t, i, p], cmap='gray')
                    axes[t, 2].set_title('H(exp) policy0: {:.2f}'.format(d['H_exp_exp_obs'][t, i, p]))

                    for k in range(self.num_classes_kn):
                        axes[t, 3 + k].imshow(exp_obs_prior[t, i, k], cmap='gray')
                        axes[t, 3 + k].set_title('k: {}'.format(k))

            [ax.set_axis_off() for ax in axes.ravel()]
            self._save_fig(f, folder_name, '{}{}_n{}.png'.format(self.prefix, suffix, i))

    @visualisation_level(2)
    def plot_fb(self, d, suffix=''):
        def fb_hist(fb1, fb2, ax, title, add_legend):
            """fb1, fb2: tuple of (values, legend)"""
            ax.hist(fb1[0], bins, alpha=0.5, label=fb1[1])
            ax.hist(fb2[0], bins, alpha=0.5, label=fb2[1])
            ax.set_title(title)
            if add_legend:
                ax.legend(loc='upper right')

        nax = self.num_classes
        ntax = self.num_glimpses
        bins = 40

        f, axes = plt.subplots(ntax, nax, figsize=(4 * nax, 4 * self.num_glimpses))

        if self.uk_label is not None:
            is_uk = (d['y'] == self.uk_label)
            fb_kn_best = d['fb'][:, ~is_uk, :].min(axis=2)  # in-shape: [T, B, hyp]
            fb_uk_best = d['fb'][:, is_uk, :].min(axis=2)
        else:
            fb_kn_best, fb_uk_best = None, None

        for t in range(ntax):
            for hyp in range(self.num_classes_kn):
                is_hyp = (d['y'] == hyp)
                if t < (ntax - 1):
                    pre = 't{}: '.format(t) if (hyp == 0) else ''
                    fb_corr  = d['fb'][t, is_hyp, hyp]
                    fb_wrong = d['fb'][t, ~is_hyp, hyp]
                else:  # last row: sum over time
                    pre = 'All t: ' if (hyp == 0) else ''
                    fb_corr  = d['fb'][:, is_hyp, hyp].sum(axis=0)
                    fb_wrong = d['fb'][:, ~is_hyp, hyp].sum(axis=0)
                fb_hist((fb_corr, 'correct hyp'),
                        (fb_wrong, 'wrong hyp'),
                        axes[t, hyp], '{}hyp: {}'.format(pre, self.lbl_map[hyp]), add_legend=(t==0))

            if self.uk_label is not None:
                # right most: best fb across hyp for kn vs uk
                if t < (ntax - 1):
                    fb_kn = fb_kn_best[t]
                    fb_uk = fb_uk_best[t]
                else:
                    fb_kn = fb_kn_best.sum(axis=0)
                    fb_uk = fb_uk_best.sum(axis=0)
                fb_hist((fb_kn, 'known'),
                        (fb_uk, 'uk'),
                        axes[t, nax - 1], 'best fb', add_legend=(t==0))

        self._save_fig(f, 'fb', '{}{}.png'.format(self.prefix, suffix))

    @visualisation_level(2)
    def plot_stateBelieves(self, d, suffix):
        # TODO: INCLUDE uk_belief and plots differentiating by known/uk
        if self.visualisation_level < 2:
            return

        ntax = self.num_glimpses
        bins = 40
        f, axes = plt.subplots(ntax, 1, figsize=(4, 4 * self.num_glimpses), squeeze=False)
        top_believes = d['state_believes'].max(axis=2)  # [T+1, B, num_classes] -> [T+1, B]

        is_corr = (d['y'] == d['clf'])
        corr = top_believes[:, is_corr]
        wrong = top_believes[:, ~is_corr]

        for t in range(ntax):
            axes[t, 0].hist(corr[t+1], bins=bins, alpha=0.5, label='corr')
            axes[t, 0].hist(wrong[t+1], bins=bins, alpha=0.5, label='wrong')
            axes[t, 0].legend(loc='upper right')
            axes[t, 0].set_title('Top believes after glimpse {}'.format(t))
            axes[t, 0].set_xlim([0, 1])

        self._save_fig(f, 'c', '{}{}.png'.format(self.prefix, suffix))
