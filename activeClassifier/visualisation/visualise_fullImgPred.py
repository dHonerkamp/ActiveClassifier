import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

logger = logging.getLogger()

from activeClassifier.visualisation.base import Visualiser, visualisation_level
from activeClassifier.tools.utility import softmax


class Visualization_fullImgPred(Visualiser):
    def __init__(self, model, FLAGS):
        super().__init__(model, FLAGS)

    @visualisation_level(1)
    def visualise(self, d, suffix='', nr_obs_overview=8, nr_obs_reconstr=5):
        nr_obs_overview = min(nr_obs_overview, self.batch_size_eff)  # batch_size_eff is set in _eval_feed() -> has to come before
        nr_obs_reconstr = min(nr_obs_reconstr, self.batch_size_eff)

        self.plot_overview(d, nr_obs_overview, suffix)
        self.plot_reconstr(d, nr_obs_reconstr, suffix)

    @visualisation_level(1)
    def intermed_plots(self, d, nr_examples, suffix='', folder_name='intermed_reconstr'):
        self.plot_reconstr(d, nr_examples, suffix, folder_name)

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

        nax = 1 + self.num_classes_kn

        reconstr = np.squeeze(d['reconstr'], -1)

        idx_examples = self._get_idx_examples(d['y'], nr_examples, replace=False)

        for i in idx_examples:
            f, axes = plt.subplots(self.num_glimpses + 1, nax, figsize=(4 * self.num_scales * nax, 4 * (self.num_glimpses + 1)))
            axes    = axes.reshape([self.num_glimpses + 1, nax])
            self._plot_img_plus_locs(axes[0, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])

            for t in range(self.num_glimpses):
                # Seen so far
                self._plot_seen(d['x'][i], d['locs'][:, i], until_t=t, ax=axes[t+1, 0])
                title = 'Label: {}, clf: {}'.format(self.lbl_map[d['y'][i]], self.lbl_map[d['clf'][i]])
                if self.uk_label is not None:
                    title += ', p(uk) post: {:.2f}'.format(d['uk_belief'][t+1, i])
                axes[t+1, 0].set_title(title)
                # next location
                decided = (d['decisions'][:t + 1, i] != -1).any()
                if not decided:
                    axes[t + 1, 0].add_patch(Rectangle(d['locs'][t, i][::-1] - self.scale_sizes[0] / 2, width=self.scale_sizes[0], height=self.scale_sizes[0], edgecolor='green', facecolor='none'))

                # reconstr for all hyp
                ranked_losses = np.argsort(d['KLdivs'][t, i, :])
                ps = softmax(-d['KLdivs'][t, i, :])
                for j, hyp in enumerate(ranked_losses):
                    axes[t+1, j + 1].imshow(reconstr[t, i, hyp], cmap='gray')
                    if d['decisions'][t, i] != -1:
                        axes[t+1, j + 1].set_title('Decision: {}'.format(d['decisions'][t, i]))
                    else:
                        c = get_title_color(d['state_believes'][t+1, i, :], hyp)
                        axes[t+1, j + 1].set_title('{}, p: {:.2f}, KL: {:.2f}\npost-c: {:.2f}, nll: {:.2f}'
                                                   .format(self.lbl_map[hyp], ps[hyp], d['KLdivs'][t, i, hyp], d['state_believes'][t+1, i, hyp], d['nll'][t, i, hyp]), color=c)

            [ax.set_axis_off() for ax in axes.ravel()]
            self._save_fig(f, folder_name, '{}{}_n{}{isuk}.png'.format(self.prefix, suffix, i,
                                                                       isuk='_uk' if (d['y'][i] == self.uk_label) else ''))
