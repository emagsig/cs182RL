# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from collections import deque
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd

def lstm_base(nlstm=128, layer_norm=False):

    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = tf.layers.flatten(X)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn


# HELPERS---------------------------

def lstm(xs, ms, s, scope, nh, init_scale=1.0):
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=utils.ortho_init(init_scale))
        wh = tf.get_variable("wh", [nh, nh*4], initializer=utils.ortho_init(init_scale))
        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c*(1-m)
        h = h*(1-m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s


def _ln(x, g, b, e=1e-5, axes=[1]):
    u, s = tf.nn.moments(x, axes=axes, keep_dims=True)
    x = (x-u)/tf.sqrt(s+e)
    x = x*g+b
    return x

def lnlstm(xs, ms, s, scope, nh, init_scale=1.0):
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=utils.ortho_init(init_scale))
        gx = tf.get_variable("gx", [nh*4], initializer=tf.constant_initializer(1.0))
        bx = tf.get_variable("bx", [nh*4], initializer=tf.constant_initializer(0.0))

        wh = tf.get_variable("wh", [nh, nh*4], initializer=utils.ortho_init(init_scale))
        gh = tf.get_variable("gh", [nh*4], initializer=tf.constant_initializer(1.0))
        bh = tf.get_variable("bh", [nh*4], initializer=tf.constant_initializer(0.0))

        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

        gc = tf.get_variable("gc", [nh], initializer=tf.constant_initializer(1.0))
        bc = tf.get_variable("bc", [nh], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c*(1-m)
        h = h*(1-m)
        z = _ln(tf.matmul(x, wx), gx, bx) + _ln(tf.matmul(h, wh), gh, bh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(_ln(c, gc, bc))
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s