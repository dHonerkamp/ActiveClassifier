import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

logger = logging.getLogger()

from activeClassifier.visualisation.base import Visualiser, visualisation_level
from activeClassifier.tools.utility import softmax

# annoying UserWarning from plt.imshow in _glimpse_patches_until_t()
import warnings
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module=r'.*matplotlib'
)


class Visualization_predRSSM(Visualiser):
    def __init__(self, model, FLAGS):
        super().__init__(model, FLAGS)

        self.num_policies = model.n_policies
        self.size_z = FLAGS.size_z
        self.planner = FLAGS.planner
        self.use_pixel_obs_FE = FLAGS.use_pixel_obs_FE
        self.rnd_first_glimpse = FLAGS.rnd_first_glimpse
        self.rnn_cell = FLAGS.rnn_cell

    @visualisation_level(1)
    def visualise(self, d, suffix='', nr_obs_overview=8, nr_obs_reconstr=5):
        nr_obs_overview = min(nr_obs_overview, self.batch_size_eff)  # batch_size_eff is set in _eval_feed() -> has to come before
        nr_obs_reconstr = min(nr_obs_reconstr, self.batch_size_eff)
        nr_obs_FE = min(3, self.batch_size_eff)

        self.plot_overview(d, nr_obs_overview, suffix)
        self.plot_reconstr(d, nr_obs_reconstr, suffix)
        self.plot_reconstr_patches(d, nr_obs_reconstr, suffix)

        # moved to batch-wise:
        # self.plot_stateBelieves(d, suffix)
        # self.plot_fb(d, prefix)
        if (self.planner == 'ActInf') & (d['epoch'] >= self.pre_train_epochs):
            # self.plot_planning(d, nr_examples=nr_obs_reconstr)
            self.plot_planning_patches(d, nr_examples=nr_obs_reconstr)
            self.plot_FE(d, nr_obs_FE, suffix)

    @visualisation_level(1)
    def intermed_plots(self, d, nr_examples, suffix='', folder_name='rnd_loc_eval'):
        self.plot_planning_patches(d, nr_examples, suffix, folder_name)

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

        gl = self._glimpse_reshp(d['glimpse'])  # [T, B, scale[0], scales*scale[0]]
        gl_post = self._glimpse_reshp(d['reconstr_posterior'])  # [T, B, scale[0], scales*scale[0]]
        gl_preds = self._glimpse_reshp(d['reconstr_prior'])  # [T, B, hyp, scale[0], scales*scale[0]]

        idx_examples = self._get_idx_examples(d['y'], nr_examples, replace=False)

        for i in idx_examples:
            f, axes = plt.subplots(self.num_glimpses + 1, nax, figsize=(4 * self.num_scales * nax, 4 * (self.num_glimpses + 1)))
            axes    = axes.reshape([self.num_glimpses + 1, nax])
            self._plot_img_plus_locs(axes[0, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])

            for t in range(self.num_glimpses):
                # true glimpse
                axes[t+1, 0].imshow(gl[t, i], **self.im_show_kwargs)
                title = 'Label: {}, clf: {}'.format(self.lbl_map[d['y'][i]], self.lbl_map[d['clf'][i]])
                if self.uk_label is not None:
                    title += ', p(uk) post: {:.2f}'.format(d['uk_belief'][t + 1, i])
                axes[t+1, 0].set_title(title)
                # posterior
                axes[t+1, 1].imshow(gl_post[t, i], **self.im_show_kwargs)
                axes[t+1, 1].set_title('Posterior, nll: {:.2f}'.format(d['nll_posterior'][t, i]))
                # prior for all classes
                ranked_losses = np.argsort(d['KLdivs'][t, i, :])
                ps = softmax(-d['KLdivs'][t, i, :])
                for j, hyp in enumerate(ranked_losses):
                    axes[t+1, j+2].imshow(gl_preds[t, i, hyp], **self.im_show_kwargs)
                    if d['decisions'][t, i] != -1:
                        axes[t+1, j + 2].set_title('Decision: {}'.format(d['decisions'][t, i]))
                    else:
                        c = get_title_color(d['state_believes'][t+1, i, :], hyp)
                        axes[t+1, j + 2].set_title('{}, p: {:.2f}, KL: {:.2f}, post-c: {:.2f}'.format(self.lbl_map[hyp], ps[hyp], d['KLdivs'][t, i, hyp], d['state_believes'][t+1, i, hyp]), color=c)

            [ax.set_axis_off() for ax in axes.ravel()]
            self._save_fig(f, folder_name, '{}{}_n{}{isuk}.png'.format(self.prefix, suffix, i,
                                                                       isuk='_uk' if (d['y'][i] == self.uk_label) else ''))

    @visualisation_level(1)
    def plot_reconstr_patches(self, d, nr_examples, suffix='', folder_name='reconstr_patches'):
        def get_title_color(post_believes, hyp):
            if post_believes[hyp] == post_believes.max():
                color = 'magenta'
            elif post_believes[hyp] > 0.1:
                color = 'blue'
            else:
                color = 'black'
            return color

        nax = 2 + self.num_classes_kn

        gl = self._glimpse_reshp(d['glimpse'])  # [T, B, scale[0], scales*scale[0]]
        gl_post = self._glimpse_reshp(d['reconstr_posterior'])  # [T, B, scale[0], scales*scale[0]]
        gl_preds = self._glimpse_reshp(d['reconstr_prior'])  # [T, B, hyp, scale[0], scales*scale[0]]

        idx_examples = self._get_idx_examples(d['y'], nr_examples, replace=False)

        for i in idx_examples:
            f, axes = plt.subplots(self.num_glimpses, nax, figsize=(4 * self.num_scales * nax, 4 * (self.num_glimpses + 1)))
            axes = axes.reshape([self.num_glimpses, nax])
            self._plot_img_plus_locs(axes[0, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])

            # rank hypotheses by final believes
            T = np.argmax(d['decisions'][:, i])  # all non-decisions are -1
            ranked_hyp = np.argsort(-d['state_believes'][T, i, :])

            for t in range(self.num_glimpses - 1):
                # true glimpses up until and including t
                self._plot_seen(d['x'][i], d['locs'][:, i], until_t=min(t + 1, self.num_glimpses), ax=axes[t + 1, 0])
                title = 'Label: {}, clf: {}'.format(self.lbl_map[d['y'][i]], self.lbl_map[d['clf'][i]])
                if self.uk_label is not None:
                    title += ', p(uk) post: {:.2f}'.format(d['uk_belief'][t + 1, i])
                axes[t + 1, 0].set_title(title)
                # posterior
                self._glimpse_patches_until_t(t+1, gl[:, i], gl_post[:, i], d['locs'][:, i], axes[t + 1, 1])
                axes[t + 1, 1].set_title('Posterior, nll: {:.2f}'.format(d['nll_posterior'][t, i]))
                # prior for all classes
                ranks_overall = np.argsort(-d['state_believes'][t, i, :]).tolist()
                ranks_kl = np.argsort(d['KLdivs'][t, i, :]).tolist()
                ps_kl = softmax(-d['KLdivs'][t, i, :])
                for j, hyp in enumerate(ranked_hyp):
                    self._glimpse_patches_until_t(min(t + 1, self.num_glimpses), gl[:, i], gl_preds[:, i, hyp], d['locs'][:, i], axes[t + 1, j + 2])
                    if d['decisions'][t, i] != -1:
                        axes[t + 1, j + 2].set_title('Decision: {}'.format(d['decisions'][t, i]))
                    else:
                        c = get_title_color(d['state_believes'][min(t + 1, self.num_glimpses), i, :], hyp)
                        axes[t + 1, j + 2].set_title('{}: tot. rank pre: {}, kl rank: {}\nsftmx(KL): {:.2f}, KL: {:.2f}, post-c: {:.2f}'.format(
                                                            self.lbl_map[hyp], ranks_overall.index(hyp), ranks_kl.index(hyp),
                                                            ps_kl[hyp], d['KLdivs'][t, i, hyp], d['state_believes'][t + 1, i, hyp]),
                                                     color=c)

            [(ax.set_xticks([]), ax.set_yticks([]), ax.set_ylim([self.img_shape[0] - 1, 0]), ax.set_xlim([0, self.img_shape[1] - 1])) for ax in axes.ravel()]
            [ax.set_axis_off() for ax in axes[0].ravel()]
            self._save_fig(f, folder_name, '{}{}_n{}{isuk}.png'.format(self.prefix, suffix, i,
                                                                       isuk='_uk' if (d['y'][i] == self.uk_label) else ''))

    def _stick_glimpse_onto_canvas(self, glimpse, loc):
        img_y, img_x = self.img_shape[:2]
        loc_y, loc_x = loc
        half_width = self.scale_sizes[0] / 2
        assert len(self.scale_sizes) == 1, 'Not adjusted for multiple scales yet'

        # Adjust glimpse if going over the edge
        y_overlap_left = -int(min(round(loc_y - half_width), 0))
        y_overlap_right = int(img_y - round(loc_y + half_width)) if ((round(loc_y + half_width) - img_y) > 0) else None
        x_overlap_left = -int(min(round(loc_x - half_width), 0))
        x_overlap_right = int(img_x - round(loc_x + half_width)) if ((round(loc_x + half_width) - img_x) > 0) else None
        glimpse = glimpse[y_overlap_left : y_overlap_right,
                          x_overlap_left : x_overlap_right]

        # Boundaries of the glimpse
        x_boundry_left  = int(max(round(loc_x - half_width), 0))
        x_boundry_right = int(min(round(loc_x + half_width), img_x))
        y_boundry_left  = int(max(round(loc_y - half_width), 0))
        y_boundry_right = int(min(round(loc_y + half_width), img_y))

        # Pad up to canvas size
        if self.img_shape[2] == 1:
            glimpse_padded = np.pad(glimpse, [(y_boundry_left, img_y - y_boundry_right),
                                              (x_boundry_left, img_x - x_boundry_right)],
                                    mode='constant')
        else:
            glimpse_padded = np.pad(glimpse, [(y_boundry_left, img_y - y_boundry_right),
                                              (x_boundry_left, img_x - x_boundry_right),
                                              (0, 0)],
                                    mode='constant')
        assert glimpse_padded.shape == tuple(self.img_shape_squeezed)
        return glimpse_padded

    def _glimpse_patches_until_t(self, until_t, true_glimpses, glimpses, locs, ax):
        """Plot the true_glimpses[:until_t - 2] & glimpses[until_t - 1] onto a canvas of shape img_shape, with the latest glimpses overlapping older ones (important for predictions)"""
        ix, iy = np.meshgrid(np.arange(self.img_shape[0]), np.arange(self.img_shape[1]))
        half_width = self.scale_sizes[0] / 2
        seen = np.zeros(self.img_shape[:2], np.bool)
        glimpse_padded = np.zeros(self.img_shape_squeezed)

        for t in range(until_t):
            loc = locs[t, :]
            y_boundry = [loc[0] - half_width, loc[0] + half_width]
            x_boundry = [loc[1] - half_width, loc[1] + half_width]
            new = (ix >= round(x_boundry[0])) & (ix < round(x_boundry[1])) & (iy >= round(y_boundry[0])) & (iy < round(y_boundry[1]))
            seen[new] = True

            input = glimpses if (t == until_t - 1) else true_glimpses
            new_glimpse_padded = self._stick_glimpse_onto_canvas(input[t], locs[t])
            glimpse_padded = np.where(new, new_glimpse_padded, glimpse_padded)

        glimpse_padded_seen = self._mask_unseen(glimpse_padded, seen)
        ax.imshow(glimpse_padded_seen, **self.im_show_kwargs)
        half_pixel = 0.5 if (self.scale_sizes[0] % 2 == 0) else 0  # glimpses are rounded to pixel values do the same for the rectangle to make it fit nicely
        ax.add_patch(Rectangle(np.round(locs[until_t - 1, ::-1] - half_width) - half_pixel, width=self.scale_sizes[0], height=self.scale_sizes[0], edgecolor='green', facecolor='none'))

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
    #                 axes[t, k].imshow(d['x'][i].reshape(self.img_shape_squeezed), **self.im_show_kwargs)
    #                 axes[t, k].scatter(locs[1], locs[0], marker='x', facecolors=color, linewidth=2.5, s=0.25 * (5 * 8 * 24))
    #                 axes[t, k].add_patch(Rectangle(locs[::-1] - self.scale_sizes[0] / 2, width=self.scale_sizes[0], height=self.scale_sizes[0], edgecolor=color, facecolor='none', linewidth=2.5))
    #                 axes[t, k].set_title('G: {:.2f}, H_: {:.2f}, exp_H: {:.2f}, G_dec: {:.2f}'.format(d['G'][t, i, k], d['H_exp_exp_obs'][t, i, k], d['exp_H'][t, i, k], d['G'][t, i, -1]))
    #
    #                 # ranked_hyp = np.argsort(d['state_believes'][t, i, :])
    #                 # for j, hyp in enumerate(ranked_hyp[::-1]):
    #                 #     # axes[t, j + 2].imshow(exp_obs[t, i, k, hyp], **self.im_show_kwargs)
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

                    axes[t, i].imshow(d['x'][i].reshape(self.img_shape_squeezed), **self.im_show_kwargs)
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
        nax_y = self.num_glimpses if self.rnd_first_glimpse else self.num_glimpses + 1

        f, axes = plt.subplots(nax_y, nax_x, figsize=(8 * self.num_scales * nax_x, 4 * nax_y), squeeze=False)
        frames_cmap = matplotlib.cm.get_cmap('bwr')
        frames_color = frames_cmap(np.linspace(1, 0, self.num_policies))
        for i in range(nr_examples):
            # if first glimpse is random, plot overview in its spot. O/w create an additional plot
            self._plot_img_plus_locs(axes[0, i], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])
            if self.rnd_first_glimpse:
                start_t = 1
                axes[0, i].set_title('t: {}, random policy, lbl: {}, clf: {}'.format(0, d['y'][i], d['clf'][i]))
            else:
                start_t = 0
                axes[0, i].set_title('Lbl: {}, clf: {}'.format(d['y'][i], d['clf'][i]))

            for ax, t in enumerate(range(start_t, self.num_glimpses)):
                ax += 1
                # plot patches seen until now
                self._plot_seen(d['x'][i], d['locs'][:, i], until_t=t, ax=axes[ax, i])

                # add current believes to legend
                ranked_believes = np.argsort(- d['state_believes'][t, i, :])
                lbl = 'hyp: ' + ', '.join('{} ({:.2f})'.format(j,  d['state_believes'][t, i, j]) for j in ranked_believes[:5])
                axes[ax, i].add_patch(Rectangle((0, 0), width=0.1, height=0.1, linewidth=0, color='white', label=lbl))

                decided = (d['decisions'][:t+1, i] != -1).any()
                if decided:
                    axes[ax, i].set_title('t: {}, decision - no new glimpse'.format(t))
                else:
                    selected = [j for j, arr in enumerate(d['potential_actions'][t, i, :]) if (arr == d['locs'][t, i]).all()]
                    axes[ax, i].set_title('t: {}, selected policy: {}'.format(t, selected[0]))

                    # plot rectangles for evaluated next locations
                    ranked_policies = np.argsort(- d['G'][t, i, :-1])
                    for iii, k in enumerate(ranked_policies):
                        # potential location under evaluation
                        locs = d['potential_actions'][t, i, k]
                        correct = np.all((locs == d['locs'][t, i, :]))

                        lbl = '{}: G: {:.2f}, H(exp): {:.2f}, E(H): {:.2f}, G_dec: {:.2f}'.format(k, d['G'][t, i, k], d['H_exp_exp_obs'][t, i, k], d['exp_H'][t, i, k], d['G'][t, i, -1])
                        axes[ax, i].add_patch(Rectangle(locs[::-1] - self.scale_sizes[0] / 2,
                                                           width=self.scale_sizes[0], height=self.scale_sizes[0],
                                                           edgecolor=frames_color[iii], facecolor='none', linewidth=1.5, label=lbl))
                        if correct:
                            axes[ax, i].scatter(locs[1], locs[0], marker='x', facecolors=frames_color[iii], linewidth=1.5, s=0.25 * (5 * 8 * 24))

                # place legend next to plot
                chartBox = axes[ax, i].get_position()
                axes[ax, i].set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
                axes[ax, i].legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)

                if decided:  # set all following axes off and stop
                    [axes[ttt, i].set_axis_off() for ttt in range(ax+1, nax_y)]
                    break

        [(ax.set_xticks([]), ax.set_yticks([]), ax.set_ylim([self.img_shape[0] - 1, 0]), ax.set_xlim([0, self.img_shape[1] - 1]))  for ax in axes.ravel()]
        self._save_fig(f, folder_name, '{}{}.png'.format(self.prefix, suffix))

    @visualisation_level(2)
    def plot_FE(self, d, nr_examples, suffix='', folder_name='FE'):
        if self.rnn_cell.startswith('Conv') and not self.use_pixel_obs_FE:
            logging.debug('Skip FE plots for convLSTM. Shapes for z not defined')
            # TODO: adjust size_z to not come from FLAGS but from VAEEncoder.output_shape_flat
            return
        # T x [True glimpse, posterior, exp_exp_obs, exp_obs...]
        nax_x = 3 + self.num_classes_kn
        nax_y = self.num_glimpses

        gl = self._glimpse_reshp(d['glimpse'])  # [T, B, scale[0], scales*scale[0]]

        if self.use_pixel_obs_FE:
            posterior = self._glimpse_reshp(d['reconstr_posterior'])
            exp_exp_obs = self._glimpse_reshp(d['exp_exp_obs'])
            exp_obs_prior = self._glimpse_reshp(d['reconstr_prior'])
        else:
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

            posterior = np.reshape(d['z_post'], [self.num_glimpses, self.batch_size_eff] + shp)
            exp_exp_obs = np.reshape(d['exp_exp_obs'], [self.num_glimpses, self.batch_size_eff, self.num_policies] + shp)
            exp_obs_prior = np.reshape(d['selected_exp_obs_enc'], [self.num_glimpses, self.batch_size_eff, self.num_classes_kn] + shp)

        for i in range(nr_examples):
            f, axes = plt.subplots(nax_y, nax_x, figsize=(4 * self.num_scales * nax_x, 4 * nax_y), squeeze=False)
            for t in range(self.num_glimpses):
                if t == 0:
                    self._plot_img_plus_locs(axes[t, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])
                else:
                    axes[t, 0].imshow(gl[t, i], **self.im_show_kwargs)
                    axes[t, 0].set_title('t: {}'.format(t))

                    axes[t, 1].imshow(posterior[t, i], **self.im_show_kwargs)
                    axes[t, 1].set_title('posterior')

                    p = d['selected_action_idx'][t, i]
                    axes[t, 2].imshow(exp_exp_obs[t, i, p], **self.im_show_kwargs)
                    axes[t, 2].set_title('H(exp) policy0: {:.2f}'.format(d['H_exp_exp_obs'][t, i, p]))

                    ranked_believes = np.argsort(- d['state_believes'][t, i, :])
                    for k in ranked_believes:
                        axes[t, 3 + k].imshow(exp_obs_prior[t, i, k], **self.im_show_kwargs)
                        axes[t, 3 + k].set_title('k: {}, p: {:.2f}'.format(k, d['state_believes'][t, i, k]))

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
        ntax = self.num_glimpses - 1
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
                if t < self.num_glimpses:
                    pre = 't{}: '.format(t) if (hyp == 0) else ''
                    fb_corr  = d['fb'][t, is_hyp, hyp]
                    fb_wrong = d['fb'][t, ~is_hyp, hyp]
                else:  # last row: sum over time
                    break
                    # pre = 'All t: ' if (hyp == 0) else ''
                    # fb_corr  = d['fb'][:, is_hyp, hyp].sum(axis=0)
                    # fb_wrong = d['fb'][:, ~is_hyp, hyp].sum(axis=0)
                fb_hist((fb_corr, 'correct hyp'),
                        (fb_wrong, 'wrong hyp'),
                        axes[t, hyp], '{}hyp: {}'.format(pre, self.lbl_map[hyp]), add_legend=(t==0))

            if self.uk_label is not None:
                # right most: best fb across hyp for kn vs uk
                if t < self.num_glimpses:
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

        ntax = self.num_glimpses
        bins = 40
        f, axes = plt.subplots(ntax, 1, figsize=(4, 4 * self.num_glimpses), squeeze=False)
        top_believes = d['state_believes'].max(axis=2)  # [T+1, B, num_classes] -> [T+1, B]
        top_believes_class = d['state_believes'].argmax(axis=2)  # [T+1, B, num_classes] -> [T+1, B]

        is_corr = (top_believes_class == d['y'][np.newaxis, :])
        corr = np.ma.masked_array(top_believes, mask=~is_corr)
        wrong = np.ma.masked_array(top_believes, mask=is_corr)

        for t in range(ntax):
            if corr[t+1].mask.any():
                axes[t, 0].hist(corr[t+1].compressed(), bins=bins, alpha=0.5, label='corr')
            if wrong[t+1].mask.any():
                axes[t, 0].hist(wrong[t+1].compressed(), bins=bins, alpha=0.5, label='wrong')
            axes[t, 0].legend(loc='upper right')
            axes[t, 0].set_title('Top believes after glimpse {}'.format(t+1))
            axes[t, 0].set_xlim([0, 1])

        self._save_fig(f, 'c', '{}{}.png'.format(self.prefix, suffix))
