import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from activeClassifier.visualisation.base import Visualiser

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Visualization_ActCl(Visualiser):
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
                      'glimpse_nll'     : model.glimpse_nlls_posterior,
                      'glimpse'         : model.obs,
                      'glimpse_reconstr': model.glimpse_reconstr,
                      }

        if self.visualisation_level > 0:
            folders = ['glimpses', 'reconstr']
            if self.visualisation_level > 1:
                if (self.planner != 'RL'):
                    folders.append('planning')

                    self.fetch.update({'exp_exp_obs' : model.exp_exp_obs,
                                       'exp_obs': model.exp_obs,
                                       'H_exp_exp_obs' : model.H_exp_exp_obs,
                                       'exp_H' : model.exp_H,
                                       'potential_actions' : model.potential_actions,})

            for f in folders:
                os.makedirs(os.path.join(FLAGS.path, f), exist_ok=True)



    def visualise(self, sess, feed, suffix=None, nr_obs_overview=8, nr_obs_reconstr=5):
        if self.visualisation_level > 0:
            d = self._eval_feed(sess, feed)

            prefix = 'epoch{}_phase{}'.format(d['epoch'], d['phase'])
            if suffix:
                prefix += '_' + suffix
            self.plot_overview(d, prefix, nr_obs_overview)
            self.plot_reconstr(d, prefix, nr_obs_reconstr)

            if self.visualisation_level > 1:
                if (self.planner != 'RL') & (d['epoch'] >= self.pre_train_epochs):
                    self.plot_planning(d, prefix, nr_examples=1)  # one plot for each policy


    def plot_reconstr(self, d, prefix, nr_examples, folder_name='reconstr'):
        nr_examples = min(nr_examples, self.batch_size_eff)
        nax = 1 + self.num_classes

        gl = self._glimpse_reshp(d['glimpse'])  # [T, B, scale[0], scales*scale[0]]
        gl_preds = self._glimpse_reshp(d['glimpse_reconstr'])  # [T, B, hyp, scale[0], scales*scale[0]]

        for i in range(nr_examples):
            f, axes = plt.subplots(self.num_glimpses + 1, nax, figsize=(4 * self.num_scales * nax, 4 * (self.num_glimpses + 1)))
            axes    = axes.reshape([self.num_glimpses + 1, nax])
            self._plot_img_plus_locs(axes[0, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])

            for t in range(self.num_glimpses):
                axes[t+1, 0].imshow(gl[t, i], cmap='gray')
                axes[t+1, 0].set_title('Label: {}, clf: {}'.format(d['y'][i], d['clf'][i]))

                ranked_nll = np.argsort(d['glimpse_nll'][t, i, :])
                ps = softmax(-d['glimpse_nll'][t, i, :])
                for j, hyp in enumerate(ranked_nll):
                    axes[t+1, j+1].imshow(gl_preds[t, i, hyp], cmap='gray')
                    axes[t+1, j+1].set_title('{}, post-c: {:.2f}, nll: {:.2f}, p: {:.2f}'.format(hyp, d['state_believes'][t, i, hyp], d['glimpse_nll'][t, i, hyp], ps[hyp]))

            # plt.setp(axes, xticks=[], yticks=[])
            [ax.set_axis_off() for ax in axes.ravel()]
            f.tight_layout()
            f.savefig('{}/{}/{}_{}_n{}.png'.format(self.path, folder_name, d['step'], prefix, i), bbox_inches='tight')
            plt.close(f)

    def plot_planning(self, d, prefix, nr_examples, folder_name='planning'):
        # T x [True glimpse, exp_exp_obs, exp_obs...]
        nax_x = 2 + self.num_classes
        nax_y = self.num_glimpses + 1

        exp_exp_obs = self._glimpse_reshp(d['exp_exp_obs'])  # [T, B, n_policies, scale[0], scales*scale[0]]
        exp_obs = self._glimpse_reshp(d['exp_obs'])  # [T, B, n_policies, num_classes, scale[0], scales*scale[0]]

        for i in range(nr_examples):
            for k in range(self.num_policies):
                f, axes = plt.subplots(nax_y, nax_x, figsize=(5 * self.num_scales * nax_x, 5 * nax_y))
                axes = axes.reshape([nax_y, nax_x])
                self._plot_img_plus_locs(axes[0, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])

                for t in range(self.num_glimpses):
                    # potential location under evaluation
                    locs = d['potential_actions'][t, i, k]
                    axes[t + 1, 0].imshow(d['x'][i].reshape(self.img_shape_squeezed), cmap='gray')
                    axes[t + 1, 0].scatter(locs[1], locs[0], marker='x', facecolors='cyan', linewidth=2.5, s=0.25 * (5 * 8 * 24))
                    axes[t + 1, 0].add_patch(Rectangle(locs[::-1] - self.scale_sizes[0] / 2, width=self.scale_sizes[0], height=self.scale_sizes[0], edgecolor='cyan', facecolor='none', linewidth=2.5))
                    axes[t + 1, 0].set_title('Label: {}, clf: {}'.format(d['y'][i], d['clf'][i]))

                    axes[t + 1, 1].imshow(exp_exp_obs[t, i, k], cmap='gray')
                    axes[t + 1, 1].set_title('G: {:.2f}, H_: {:.2f}, exp_H: {:.2f}, extr: {:.2f}'.format(d['G'][t, i, k], d['H_exp_exp_obs'][t, i, k], d['exp_H'][t, i, k], d['G'][t, i, -1]))

                    ranked_hyp = np.argsort(d['state_believes'][t, i, :])
                    for j, hyp in enumerate(ranked_hyp[::-1]):
                        axes[t + 1, j + 2].imshow(exp_obs[t, i, k, hyp], cmap='gray')
                        axes[t + 1, j + 2].set_title('Hyp: {}, prob: {:.2f}'.format(hyp, d['state_believes'][t, i, hyp]))

                [ax.set_axis_off() for ax in axes.ravel()]
                f.tight_layout()
                f.savefig('{}/{}/{}_{}_n{}_k{}.png'.format(self.path, folder_name, d['step'], prefix, i, k), bbox_inches='tight')
                plt.close(f)
