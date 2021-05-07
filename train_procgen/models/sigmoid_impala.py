""" USE CPU """
# import tensorflow as tf
""" USE GPU """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd


def sigmoid_impala_model(unscaled_images, depths=[16,32,32,32]):


    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str
    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.sigmoid(inputs)
        out = conv_layer(out, depth)
        out = tf.nn.sigmoid(out)
        out = conv_layer(out, depth)
        out = tf.nn.dropout(out, .6)
        
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    # change to sigmoid
    # out = tf.nn.sigmoid(out)
    # out = tf.layers.dense(out, 256, activation=tf.nn.sigmoid, name='layer_' + get_layer_num_str())

    return out