import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from visualisation.base import Base

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
                      }

        if self.visualisation_level > 0:
            folders = ['glimpses', 'reconstr']

            if FLAGS.uk_label is not None:
                self.fetch.update({'uk_belief': model.uk_belief, })

            if self.visualisation_level > 1:
                folders.append('fb')
                folders.append('c')
            #     if (self.planner != 'RL'):
            #         folders.append('planning')
            #
            #         self.fetch.update({'exp_exp_obs' : model.exp_exp_obs,
            #                            'exp_obs': model.exp_obs,
            #                            'H_exp_exp_obs' : model.H_exp_exp_obs,
            #                            'exp_H' : model.exp_H,
            #                            'potential_actions' : model.potential_actions,})

            for f in folders:
                os.makedirs(os.path.join(FLAGS.path, f), exist_ok=True)

    def visualise(self, sess, feed, suffix='', nr_obs_overview=8, nr_obs_reconstr=5):
        if self.visualisation_level > 0:
            d = self._eval_feed(sess, feed)

            self.prefix = 's{}_e{}_ph{}'.format(d['step'], d['epoch'], d['phase'])
            self.plot_overview(d, nr_obs_overview, suffix)
            self.plot_reconstr(d, nr_obs_reconstr, suffix)

            if self.visualisation_level > 1:
                self.plot_stateBelieves(d, suffix)
            #     self.plot_fb(d, prefix)
            #     if (self.planner != 'RL') & (d['epoch'] >= self.pre_train_epochs):
            #         self.plot_planning(d, prefix, nr_examples=1)  # one plot for each policy

    def plot_reconstr(self, d, nr_examples, suffix='', folder_name='reconstr'):
        def get_title_color(post_believes, hyp):
            if post_believes[hyp] == post_believes.max():
                color = 'magenta'
            elif post_believes[hyp] > 0.1:
                color = 'blue'
            else:
                color = 'black'
            return color

        nr_examples = min(nr_examples, self.batch_size_eff)
        nax = 2 + self.num_classes_kn

        gl = self._scale_reshp(d['glimpse'])  # [T, B, scale[0], scales*scale[0]]
        gl_post = self._scale_reshp(d['reconstr_posterior'])  # [T, B, scale[0], scales*scale[0]]
        gl_preds = self._scale_reshp(d['reconstr_prior'])  # [T, B, hyp, scale[0], scales*scale[0]]

        for i in range(nr_examples):
            f, axes = plt.subplots(self.num_glimpses + 1, nax, figsize=(4 * self.num_scales * nax, 4 * (self.num_glimpses + 1)))
            axes    = axes.reshape([self.num_glimpses + 1, nax])
            self._plot_img_plus_locs(axes[0, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])

            for t in range(self.num_glimpses):
                # true glimpse
                axes[t+1, 0].imshow(gl[t, i], cmap='gray')
                title = 'Label: {}, clf: {}'.format(d['y'][i], d['clf'][i])
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
                        axes[t + 1, j + 2].set_title('Decision: {}'.format(d['decisions'][t, i]))
                    else:
                        c = get_title_color(d['state_believes'][t+1, i, :], hyp)
                        axes[t+1, j+2].set_title('{}, p: {:.2f}, KL: {:.2f}, post-c: {:.2f}'.format(self.lbl_map[hyp], ps[hyp], d['KLdivs'][t, i, hyp], d['state_believes'][t+1, i, hyp]), color=c)

            # plt.setp(axes, xticks=[], yticks=[])
            [ax.set_axis_off() for ax in axes.ravel()]
            self._save_fig(f, folder_name, '{}{}_n{}.png'.format(self.prefix, suffix, i))

    def plot_fb(self, d, suffix=''):
        def fb_hist(fb1, fb2, ax, title, add_legend):
            """fb1, fb2: tuple of (values, legend)"""
            ax.hist(fb1[0], bins, alpha=0.5, label=fb1[1])
            ax.hist(fb2[0], bins, alpha=0.5, label=fb2[1])
            ax.set_title(title)
            if add_legend:
                ax.legend(loc='upper right')

        if self.visualisation_level > 1:
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

    def plot_stateBelieves(self, d, suffix):
        # TODO: INCLUDE uk_belief and plots differentiating by known/uk
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
