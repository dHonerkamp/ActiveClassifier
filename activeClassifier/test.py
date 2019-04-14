from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from env.env import ImageForagingEnvironment
from tools.utility import Utility
from tools.tf_tools import repeat_axis
from tools.MSE_distribution import MSEDistribution
from modules.policyNetwork import PolicyNetwork
from env import create_class_mapping_ukMax

def setup():
    FLAGS, unparsed = Utility.parse_arg()
    Utility.auto_adjust_flags(FLAGS)

    Utility.set_seeds()

    train_data, valid_data, test_data = get_data(FLAGS)

    # steup environment
    env = ImageForagingEnvironment(FLAGS)

    return FLAGS, train_data, valid_data, test_data, env

def env_step(env):
    policyNet = PolicyNetwork(FLAGS, env.batch_sz)
    next_action, next_action_mean = policyNet.inital_loc()

    next_observation, feedback, done = env.step(next_action, decision=tf.zeros([env.batch_sz], tf.int32))
    # glimpse_composed = env.composed_glimpse(FLAGS, tf.expand_dims(next_observation, 0))

    with tf.Session() as sess:
        train_handle, valid_handle, test_handle = env.intialise(train_data, valid_data, test_data, sess)

        feed = {
                # model.is_training: True,
                env.handle       : train_handle,
                # model.true_label : False,
                env.MC_samples   : FLAGS.MC_samples}

        next_observation, feedback, done = sess.run([next_observation, feedback, done], feed_dict=feed)
        # glimpse = sess.run([glimpse_composed], feed_dict=feed)

        glimpses = next_observation.reshape([-1, 8, 8])
        plt.imshow(glimpses[10])
        plt.show()

        print(feedback)
        print(done)


def hyp_tiling():
    num_classes = 5
    hyp = tf.tile(tf.one_hot(tf.range(num_classes), depth=num_classes),
                  [2, 1])

    with tf.Session() as sess:
        out = sess.run(hyp)
        print(out)

        assert (out == [[1., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 1.],
                        [1., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 1.]]).all()

def hyp_tiling2():
    num_classes = 5
    hyp = repeat_axis(tf.one_hot(tf.range(num_classes), depth=num_classes),
                      axis=0,
                      repeats=2)

    with tf.Session() as sess:
        out = sess.run(hyp)
        assert (out == [[1., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0.],
                        [0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 0., 1., 0.],
                        [0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 1.],
                        [0., 0., 0., 0., 1.]]).all()



def z_tiling():
    batch_sz = 3
    size_z = 3
    n_policies = 2
    hyp = 4

    dummy = np.tile(np.arange(batch_sz)[:, np.newaxis, np.newaxis], [1, hyp, size_z])
    assert (dummy == [[[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],
                     [[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]],
                     [[2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2]]]).all()
    assert dummy.shape == (batch_sz, hyp, size_z)

    repeated = repeat_axis(tf.constant(dummy), axis=0, repeats=n_policies)  # [B, hyp, z] -> [[B] * n_policies, hyp, z]
    tiled = tf.reshape(repeated, [batch_sz * hyp * n_policies, size_z])  # [[B] * n_policies, hyp, z] -> [[B * hyp] * n_policies, z]
    un_tiled = tf.reshape(tiled, [batch_sz, n_policies, hyp, size_z])

    with tf.Session() as sess:
        rep, til, un_til = sess.run([repeated, tiled, un_tiled])
        assert rep.shape == (n_policies * batch_sz, hyp, size_z)
        assert (rep == [[[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]],
                        [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]],
                        [[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]],
                        [[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]],
                        [[2, 2, 2],
                         [2, 2, 2],
                         [2, 2, 2],
                         [2, 2, 2]],
                        [[2, 2, 2],
                         [2, 2, 2],
                         [2, 2, 2],
                         [2, 2, 2]]]).all()

        # per batch-obs: for each of hyp=4, n_poicies=2 times tiled below each other
        assert til.shape == (n_policies * batch_sz * hyp, size_z)
        assert (til == [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],

                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],

                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],

                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],

                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],

                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2]]).all()

        assert un_til.shape == (batch_sz, n_policies, hyp, size_z)
        assert (un_til == [[[[0, 0, 0],  # [B, n_policies, hyp, size_z]
                             [0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]],
                            [[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]]],
                           [[[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]],
                            [[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]]],
                           [[[2, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2]],
                            [[2, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2],
                             [2, 2, 2]]]]).all()

def z_indexing():
    hyp = 3
    batch_sz = 5
    size_z = 2
    dummy = np.tile(np.arange(hyp)[np.newaxis, :, np.newaxis], [batch_sz, 1, size_z])

    k = 1
    coords = tf.stack(tf.meshgrid(tf.range(batch_sz)) + [tf.fill([batch_sz], k)], axis=1)
    class_conditional_z = tf.gather_nd(dummy, coords)

    with tf.Session() as sess:
        class_cond_z = sess.run(class_conditional_z)
        assert (class_cond_z == [[1, 1],
                                 [1, 1],
                                 [1, 1],
                                 [1, 1],
                                 [1, 1]]).all()


def MSE_dist():
    x = tf.constant(np.random.random([10, 5]), tf.float32)
    mean = tf.constant(np.random.random([10, 5]), tf.float32)
    dist = MSEDistribution(mean)

    log_prob = dist.log_prob(x)

    with tf.Session() as sess:
        out = log_prob.eval()
        print(out.shape)


def negLabel_xent():
    y = np.array([-1, -1, 0, 1])
    logits = np.array([[0.5, 0.4, 0.3],
                       [0.1, 0.2, 0.4],
                       [0.1, 0.6, 0.3],
                       [0.1, 0.6, 0.3]])

    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    with tf.Session() as sess:
        out = xent.eval()
        print(out)


def uk_mapping():
    num_clases = 10
    uks = [3, 6, 8]
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3])

    mapping0, uk_class = create_class_mapping_uk0(num_clases, uks)
    print(mapping0, uk_class)

    out = np.vectorize(mapping0.get)(labels)
    print(out)


    mappingMax, uk_class = create_class_mapping_ukMax(num_clases, uks)
    print(mappingMax, uk_class)

    out = np.vectorize(mappingMax.get)(labels)
    print(out)


def one_hot_incl_uk():
    labels = np.array([0, 1, 2, 3, 4, 0])

    oneHot = tf.one_hot(labels, depth=4)
    with tf.Session() as sess:
        out = oneHot.eval()

        assert (out == [[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.],
                        [0., 0., 0., 0.],
                        [1., 0., 0., 0.]]).all()

if __name__ == '__main__':
    # FLAGS, train_data, valid_data, test_data, env = setup()
    #
    # prior_preferences = [0, FLAGS.prior_preference_c, -2*FLAGS.prior_preference_c]  # [prediction error, correct classification, wrong classification]
    # model = ActiveClassifier(FLAGS, env, prior_preferences)
    #
    # env_step(env)

    # hyp_tiling()
    # hyp_tiling2()
    # z_tiling()
    # z_indexing()

    # MSE_dist()

    # negLabel_xent()
    # uk_mapping()

    one_hot_incl_uk()