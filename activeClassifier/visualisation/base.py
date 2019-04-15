import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Base:
    def __init__(self, model, FLAGS):
        self.path = FLAGS.path
        self.dataset = FLAGS.dataset
        self.visualisation_level = FLAGS.visualisation_level
        self.pre_train_epochs = FLAGS.pre_train_epochs
        self.img_shape = FLAGS.img_shape
        self.img_shape_squeezed = (self.img_shape[:2] if self.img_shape[2] == 1 else self.img_shape)
        self.num_scales = len(FLAGS.scale_sizes)
        self.scale_sizes = FLAGS.scale_sizes
        self.num_classes = FLAGS.num_classes
        self.num_classes_kn = FLAGS.num_classes_kn
        self.num_policies = model.n_policies
        self.size_z = FLAGS.size_z
        self.planner = FLAGS.planner
        self.uk_label = FLAGS.uk_label
        self.lbl_map = FLAGS.class_remapping


    def visualise(self, sess, feed, suffix=None, nr_obs_overview=8, nr_obs_reconstr=5):
        pass

    def _adjust_loc(self, locs):
        # extract_glimpses: (-1,-1) is top left. 0 is y-axis, 1 is x-axis. Scale of imshow shifted by 1.
        locs = np.clip(locs, -1, 1)
        locs = (locs / 2 + 0.5) * self.img_shape[1::-1] - 1  # in img_shape y comes first, then x
        return locs

    def _get_idx_examples(self, y, nr_examples, nr_uk=2, replace=True):
        """Return the indexes of examples to plot, including nr_uk uks."""
        nr_examples = min(nr_examples, self.batch_size_eff)

        if self.uk_label is None:
            idx_examples =list(range(nr_examples))
        else:  # ensure at least nr_uk uk examples
            is_uk = (y == self.uk_label)
            idx_uk = np.arange(self.batch_size_eff)[is_uk]
            idx_kn = np.arange(self.batch_size_eff)[~is_uk]

            nr_kn = nr_examples - nr_uk if replace or (nr_examples > self.batch_size_eff) else nr_examples
            idx_examples = list(idx_kn[:nr_kn])
            idx_examples += list(idx_uk[:nr_uk])
        return idx_examples

    def _eval_feed(self, sess, feed):
        d = sess.run(self.fetch, feed_dict=feed)

        d['phase'] = d['phase'].decode("utf-8")

        d['locs'] = self._adjust_loc(d['locs'])
        if 'potential_actions' in d.keys():
            d['potential_actions'] = self._adjust_loc(d['potential_actions'])

        # update num_glimpses as it might be dynamic
        self.num_glimpses, self.batch_size_eff = d['locs'].shape[0], d['locs'].shape[1]
        return d

    def plot_overview(self, d, nr_examples, suffix=''):
        f, axes = plt.subplots(nr_examples, self.num_glimpses + 1, figsize=(6 * self.num_glimpses, 5 * nr_examples))
        axes    = axes.reshape([nr_examples, self.num_glimpses + 1])

        nr_examples = min(self.batch_size_eff, nr_examples)  # might have less examples if at the end of a batch
        idx_examples = self._get_idx_examples(d['y'], nr_examples, nr_uk=4, replace=True)
        for a, i in enumerate(idx_examples):
            self._plot_img_plus_locs(axes[a, 0], d['x'][i], d['y'][i], d['clf'][i], d['locs'][:, i, :], d['decisions'][:, i])
            for t in range(self.num_glimpses):
                self._plot_composed_glimpse(axes[a, t + 1], d['gl_composed'][t, i], d['state_believes'][t, i], d['decisions'][t, i])

        plt.setp(axes, xticks=[], yticks=[])
        self._save_fig(f, 'glimpses', '{}{}.png'.format(self.prefix, suffix))

    def _plot_img_plus_locs(self, ax, x, y, clf, locs, decisions):
        ax.imshow(x.reshape(self.img_shape_squeezed), cmap='gray')
        ax.set_title('Label: {} Classification: {}'.format(self.lbl_map[y], self.lbl_map[clf]))

        for t in range(self.num_glimpses):
            if decisions[t] != -1:  # decision taken, locations are zeroed-out
                assert (locs[t] == self._adjust_loc(np.array([0., 0.]))).all(), 'Decision indicates no further glimpses taken, but locs are not zeroed-out'
                break
            c = ('green' if clf == y else 'red')
            if t == 0:
                m = 'x'; fc = c
            else:
                m = 'o'; fc = 'none'
            # plot glimpse location
            ax.scatter(locs[t, 1], locs[t, 0], marker=m, facecolors=fc, edgecolors=c, linewidth=2.5, s=0.25 * (5 * 8 * 24))
            # connecting line
            ax.plot(locs[t - 1:t + 1, 1], locs[t - 1:t + 1, 0], linewidth=2.5, color=c)
            # rectangle around location?
            # ax.add_patch(Rectangle(locs[n,i][::-1] - FLAGS.scale_sizes[0] / 2, width=FLAGS.scale_sizes[0], height=FLAGS.scale_sizes[0], edgecolor=c, facecolor='none'))

        ax.set_ylim([self.img_shape[0] - 1, 0])
        ax.set_xlim([0, self.img_shape[1] - 1])

    def _plot_composed_glimpse(self, ax, gl_composed, belief, decision):
        ax.imshow(gl_composed.reshape(2 * [np.max(self.scale_sizes)] + [self.img_shape[-1]]).squeeze(), cmap='gray')
        top_belief = np.argmax(belief)
        ax.set_title('class {}, belief: {}, prob: {:.3f}'.format(decision, top_belief, belief[top_belief]))

    def _scale_reshp(self, x):
        shp = x.shape
        ndims = len(shp) - 1
        sc0 = self.scale_sizes[0]
        x = np.reshape(x, list(shp[:-1]) + [self.num_scales, sc0, sc0])
        # combine the num_scales into the horizontal axis
        x = np.transpose(x, list(range(ndims)) + [ndims + 1, ndims, ndims + 2])
        x = np.reshape(x, list(x.shape[:-3]) + [sc0, self.num_scales * sc0])
        return x

    def _save_fig(self, f, folder_name, name):
        f.tight_layout()
        f.savefig(os.path.join(self.path, folder_name, name), bbox_inches='tight')
        plt.close(f)
