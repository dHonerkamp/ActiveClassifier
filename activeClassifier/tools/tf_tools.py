import tensorflow as tf
import math

from tensorflow.contrib.distributions import Logistic

TINY = 1e-8


def output_into_gaussian_params(output, min_std=TINY, axis=-1):
    mu, sigma = tf.split(output, 2, axis=axis)
    sigma = tf.nn.softplus(sigma)
    sigma = tf.maximum(sigma, min_std)

    return tf.distributions.Normal(mu, sigma)


def create_MLP(specs, name_prefix=''):
    layers = []
    for i, spec in enumerate(specs):
        n, act = spec
        l = tf.layers.Dense(n, activation=act, name=name_prefix + 'fc{}'.format(i))
        layers.append(l)
    return layers


def binary_entropy(logits=None, probs=None, return_nats=True):
    if (probs is None) == (logits is None):
        raise ValueError("Must pass probs or logits, but not both.")

    if logits is not None:
        H = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.nn.sigmoid(logits), logits=logits)
    else:
        probs = tf.stack([probs, 1. - probs], axis=-1)
        H = entropy(probs=probs)

    if return_nats:# change of basis into nats
        H /= tf.log(2.)
    return H


def entropy(probs=None, logits=None, axis=-1):
  """Compute entropy over specified dimensions (i.e. assumes one-hot encoded vector)."""
  if (probs is None) == (logits is None):
      raise ValueError("Must pass probs or logits, but not both.")

  if logits is not None:
      plogp = tf.nn.softmax(logits, axis) * tf.nn.log_softmax(logits, axis)
  else:
      plogp = probs * tf.log(probs + TINY)
  return -tf.reduce_sum(plogp, axis)


def differential_entropy_normal(sigma, axis=-1):
    """Compute the differential entropy of a normal distribution. Can be negative!"""
    entr = tf.log(sigma * tf.sqrt(2 * math.pi * math.e) + TINY)
    return tf.reduce_sum(entr, axis)


def differential_entropy_diag_normal(sigma, axis=-1):
    """Compute the differential entropy of a normal distribution. Can be negative!"""
    N = tf.reduce_prod(tf.cast(tf.shape(sigma)[axis], tf.float32))
    det = tf.reduce_prod(sigma, axis)
    entr = 0.5 * tf.log(tf.sqrt(2 * math.pi * math.e)**N * det)
    return entr


def repeat_axis(x, axis, repeats):
    shp = x.shape.as_list()

    x_exp = tf.expand_dims(x, axis + 1)
    multiples = x_exp.shape.ndims * [1]
    multiples[axis + 1] = repeats
    x_tiled = tf.tile(x_exp, multiples)

    reps = -1 if shp[axis] is None else shp[axis] * repeats
    new_dims = list(shp[:axis]) + [reps] + list(shp[axis + 1:])
    assert new_dims.count(-1) + new_dims.count(None) <= 1
    return tf.reshape(x_tiled, shape=new_dims)


def calculate_gaussian_nll(predicted, actual):
    dist = tf.distributions.Normal(predicted['mu'], predicted['sigma'])
    loss = -dist.log_prob(actual)
    return tf.reduce_sum(loss, axis=-1)


def FiLM_layer(context_inputs, main_inputs, conv_input=False, name='FiLM'):
    units = main_inputs.shape[-1]
    gamma = tf.layers.dense(context_inputs, units, name=name + '_gamma')
    beta = tf.layers.dense(context_inputs, units, name=name + '_beta')
    if conv_input:
        gamma = gamma[:, tf.newaxis, tf.newaxis, :]
        beta = beta[:, tf.newaxis, tf.newaxis, :]
    return tf.nn.tanh(gamma * main_inputs + beta)


def batch_min_normalization(predErrors, axis=0, epsilon=TINY):
    """
    Only for all positive predErrors.
    Args:
        predErrors: prediction errors in shape [B, hyp]
    """
    min = tf.reduce_min(predErrors, axis=axis, keep_dims=True)  # over batch axis
    out = predErrors / (tf.stop_gradient(min) + epsilon)
    return out


def expanding_mean(new_value, old_value, time):
    """
    Args:
        new_value: this time step's value
        old_value: aggregate from last time step
        time: current time, zero-indexed
    Returns:
        mean from t = 0 : time
    """
    time_1plus = time + 1
    return (1. / time_1plus) * new_value + (time_1plus - 1.) / time_1plus * old_value


def exponential_mov_avg(tracked_var, avg_var, decay):
    return decay * avg_var + (1 - decay) * tracked_var


def pseudo_LogRelaxedBernoulli(logits, temperature, allow_nan_stats=True):
    return Logistic(loc=logits / temperature, scale=1 / temperature, allow_nan_stats=allow_nan_stats)


# def LogRelaxedBernoulli_logprob(y, logits, temperature):
#     diff = logits - temperature * y
#     return tf.log(temperature) + diff - 2 * tf.log1p(tf.exp(diff))