#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
model
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import sys
import math
import tensorflow as tf
import numpy as np

sys.path.append("../")
from network import UNetDS2, UniNetDS, ConvGRUCell
from homography_warping import *

FLAGS = tf.app.flags.FLAGS


#################################################
def inference_prob_recurrent(images, cams, depth_num, depth_start, depth_interval, is_master_gpu=True):
    """ infer depth map from mvs images and cameras """

    # dynamic gpu params
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = UniNetDS({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UniNetDS({'data': ref_image}, is_training=True, reuse=True)

    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UniNetDS({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies_Twc(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    gru1_filters = 8
    gru2_filters = 16
    gru3_filters = 32
    gru4_filters = 64

    feature_shape1 = [FLAGS.batch_size, FLAGS.max_h / 2, FLAGS.max_w / 2, gru1_filters]
    feature_shape2 = [FLAGS.batch_size, FLAGS.max_h / 4, FLAGS.max_w / 4, gru2_filters]
    feature_shape3 = [FLAGS.batch_size, FLAGS.max_h / 8, FLAGS.max_w / 8, gru3_filters]
    feature_shape4 = [FLAGS.batch_size, FLAGS.max_h / 16, FLAGS.max_w / 16, gru4_filters]

    gru_input_shape1 = [feature_shape1[1], feature_shape1[2]]
    gru_input_shape2 = [feature_shape2[1], feature_shape2[2]]
    gru_input_shape3 = [feature_shape3[1], feature_shape3[2]]
    gru_input_shape4 = [feature_shape4[1], feature_shape4[2]]

    state1 = tf.zeros([FLAGS.batch_size, feature_shape1[1], feature_shape1[2], gru1_filters])
    state2 = tf.zeros([FLAGS.batch_size, feature_shape2[1], feature_shape2[2], gru2_filters])
    state3 = tf.zeros([FLAGS.batch_size, feature_shape3[1], feature_shape3[2], gru3_filters])
    state4 = tf.zeros([FLAGS.batch_size, feature_shape4[1], feature_shape4[2], gru4_filters])

    conv_gru1 = ConvGRUCell(shape=gru_input_shape1, kernel=[3, 3], filters=gru1_filters)
    conv_gru2 = ConvGRUCell(shape=gru_input_shape2, kernel=[3, 3], filters=gru2_filters)
    conv_gru3 = ConvGRUCell(shape=gru_input_shape3, kernel=[3, 3], filters=gru3_filters)
    conv_gru4 = ConvGRUCell(shape=gru_input_shape4, kernel=[3, 3], filters=gru4_filters)

    with tf.name_scope('cost_volume_homography'):

        # forward cost volume
        depth_costs = []
        for d in range(depth_num):

            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())

            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(
                    view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                warped_view_feature = tf_transform_homography(view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost = ave_feature2 - tf.square(ave_feature)

            # U1
            conv_cost1 = tf.layers.conv2d(-cost, 16, 3, strides=(2, 2), padding='same', activation='relu',
                                          reuse=tf.AUTO_REUSE, name='conv1')  # (1, 96, 192, 16),
            conv_cost2 = tf.layers.conv2d(conv_cost1, 32, 3, strides=(2, 2), padding='same', activation='relu',
                                          reuse=tf.AUTO_REUSE, name='conv2')  # (1, 48, 96, 32),
            conv_cost3 = tf.layers.conv2d(conv_cost2, 64, 3, strides=(2, 2), padding='same', activation='relu',
                                          reuse=tf.AUTO_REUSE, name='conv3')  # (1, 24, 48, 64),

            reg_cost4, state4 = conv_gru4(conv_cost3, state4, scope='conv_gru4')  # (1, 24, 48, 64)
            up_cost3 = tf.layers.conv2d_transpose(reg_cost4, 32, 3, strides=(2, 2), padding='same', activation='relu',
                                                  reuse=tf.AUTO_REUSE, name='up_conv3')  # (1, 48, 96, 32)

            reg_cost3, state3 = conv_gru3(conv_cost2, state3, scope='conv_gru3')  # (1, 48, 96, 32)
            up_cost33 = tf.add(up_cost3, reg_cost3, name='add3')  # (1, 48, 96, 32)

            up_cost2 = tf.layers.conv2d_transpose(up_cost33, 16, 3, strides=(2, 2), padding='same', activation='relu',
                                                  reuse=tf.AUTO_REUSE, name='up_conv2')  # (1, 96, 192, 16)
            reg_cost2, state2 = conv_gru2(conv_cost1, state2, scope='conv_gru2')  # (1, 96, 192, 16)
            up_cost22 = tf.add(up_cost2, reg_cost2, name='add2')  # (1, 96, 192, 16)

            up_cost1 = tf.layers.conv2d_transpose(up_cost22, 8, 3, strides=(2, 2), padding='same', activation='relu',
                                                  reuse=tf.AUTO_REUSE, name='up_conv1')  # (1, 192, 384, 8)
            reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')  # (1, 192, 384, 8)
            up_cost11 = tf.add(up_cost1, reg_cost1, name='add2')  # (1, 192, 384, 8)

            reg_cost = tf.layers.conv2d_transpose(up_cost11, 1, 3, strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE,
                                                  name='prob_conv')  # (1, 384, 768, 1)
            depth_costs.append(reg_cost)

        prob_volume = tf.stack(depth_costs, axis=1)
        prob_volume = tf.nn.softmax(prob_volume, axis=1, name='prob_volume')

    return prob_volume


def inference_winner_take_all(images, cams, depth_num, depth_start, depth_end, is_master_gpu=True, inverse_depth=False):
    """ infer depth map from mvs images and cameras """

    if not inverse_depth:
        depth_interval = (depth_end - depth_start) / (tf.cast(depth_num, tf.float32) - 1)

    # reference image
    ref_image = tf.squeeze(tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    if is_master_gpu:
        ref_tower = UniNetDS({'data': ref_image}, is_training=True, reuse=False)
    else:
        ref_tower = UniNetDS({'data': ref_image}, is_training=True, reuse=True)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UniNetDS({'data': view_image}, is_training=True, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                                      depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies_Twc(ref_cam, view_cam, depth_num=depth_num,
                                                depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    # gru unit
    gru1_filters = 8
    gru2_filters = 16

    gru3_filters = 32
    gru4_filters = 64

    max_h = int(FLAGS.max_h * FLAGS.resize_scale)
    max_w = int(FLAGS.max_w * FLAGS.resize_scale)

    feature_shape0 = [FLAGS.batch_size, int(max_h), int(max_w), 1]
    feature_shape1 = [FLAGS.batch_size, int(max_h / 2), int(max_w / 2), gru1_filters]
    feature_shape2 = [FLAGS.batch_size, int(max_h / 4), int(max_w / 4), gru2_filters]
    feature_shape3 = [FLAGS.batch_size, int(max_h / 8), int(max_w / 8), gru3_filters]
    feature_shape4 = [FLAGS.batch_size, int(max_h / 16), int(max_w / 16), gru4_filters]

    gru_input_shape1 = [feature_shape1[1], feature_shape1[2]]
    gru_input_shape2 = [feature_shape2[1], feature_shape2[2]]
    gru_input_shape3 = [feature_shape3[1], feature_shape3[2]]
    gru_input_shape4 = [feature_shape4[1], feature_shape4[2]]

    state1 = tf.zeros([FLAGS.batch_size, feature_shape1[1], feature_shape1[2], gru1_filters])
    state2 = tf.zeros([FLAGS.batch_size, feature_shape2[1], feature_shape2[2], gru2_filters])
    state3 = tf.zeros([FLAGS.batch_size, feature_shape3[1], feature_shape3[2], gru3_filters])
    state4 = tf.zeros([FLAGS.batch_size, feature_shape4[1], feature_shape4[2], gru4_filters])

    conv_gru1 = ConvGRUCell(shape=gru_input_shape1, kernel=[3, 3], filters=gru1_filters)
    conv_gru2 = ConvGRUCell(shape=gru_input_shape2, kernel=[3, 3], filters=gru2_filters)
    conv_gru3 = ConvGRUCell(shape=gru_input_shape3, kernel=[3, 3], filters=gru3_filters)
    conv_gru4 = ConvGRUCell(shape=gru_input_shape4, kernel=[3, 3], filters=gru4_filters)

    # initialize variables
    exp_sum = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape0[1], feature_shape0[2], 1]),
        name='exp_sum', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    depth_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape0[1], feature_shape0[2], 1]),
        name='depth_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    max_prob_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape0[1], feature_shape0[2], 1]),
        name='max_prob_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    init_map = tf.zeros([FLAGS.batch_size, feature_shape0[1], feature_shape0[2], 1])

    # define winner take all loop
    def body(depth_index, state1, state2, state3, state4, depth_image, max_prob_image, exp_sum, incre):
        """Loop body."""

        # calculate cost
        ave_feature = ref_tower.get_output()
        ave_feature2 = tf.square(ref_tower.get_output())
        for view in range(0, FLAGS.view_num - 1):
            homographies = view_homographies[view]
            homographies = tf.transpose(homographies, perm=[1, 0, 2, 3])
            homography = homographies[depth_index]
            warped_view_feature = tf_transform_homography(view_towers[view].get_output(), homography)
            ave_feature = ave_feature + warped_view_feature
            ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
        ave_feature = ave_feature / FLAGS.view_num
        ave_feature2 = ave_feature2 / FLAGS.view_num
        cost = ave_feature2 - tf.square(ave_feature)
        cost.set_shape([FLAGS.batch_size, feature_shape1[1], feature_shape1[2], 16])

        # U1
        conv_cost1 = tf.layers.conv2d(-cost, 16, 3, strides=(2, 2), padding='same', activation='relu',
                                      reuse=tf.AUTO_REUSE, name='conv1')  # (1, 96, 192, 16),
        conv_cost2 = tf.layers.conv2d(conv_cost1, 32, 3, strides=(2, 2), padding='same', activation='relu',
                                      reuse=tf.AUTO_REUSE, name='conv2')  # (1, 48, 96, 32),
        conv_cost3 = tf.layers.conv2d(conv_cost2, 64, 3, strides=(2, 2), padding='same', activation='relu',
                                      reuse=tf.AUTO_REUSE, name='conv3')  # (1, 24, 48, 64),

        reg_cost4, state4 = conv_gru4(conv_cost3, state4, scope='conv_gru4')  # (1, 24, 48, 64)
        up_cost3 = tf.layers.conv2d_transpose(reg_cost4, 32, 3, strides=(2, 2), padding='same', activation='relu',
                                              reuse=tf.AUTO_REUSE, name='up_conv3')  # (1, 48, 96, 32)

        reg_cost3, state3 = conv_gru3(conv_cost2, state3, scope='conv_gru3')  # (1, 48, 96, 32)
        up_cost33 = tf.add(up_cost3, reg_cost3, name='add3')  # (1, 48, 96, 32)

        up_cost2 = tf.layers.conv2d_transpose(up_cost33, 16, 3, strides=(2, 2), padding='same', activation='relu',
                                              reuse=tf.AUTO_REUSE, name='up_conv2')  # (1, 96, 192, 16)
        reg_cost2, state2 = conv_gru2(conv_cost1, state2, scope='conv_gru2')  # (1, 96, 192, 16)
        up_cost22 = tf.add(up_cost2, reg_cost2, name='add2')  # (1, 96, 192, 16)

        up_cost1 = tf.layers.conv2d_transpose(up_cost22, 8, 3, strides=(2, 2), padding='same', activation='relu',
                                              reuse=tf.AUTO_REUSE, name='up_conv1')  # (1, 192, 384, 8)
        reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')  # (1, 192, 384, 8)
        up_cost11 = tf.add(up_cost1, reg_cost1, name='add2')  # (1, 192, 384, 8)

        reg_cost = tf.layers.conv2d_transpose(up_cost11, 1, 3, strides=(2, 2), padding='same', reuse=tf.AUTO_REUSE,
                                              name='prob_conv')  # (1, 384, 768, 1)
        prob = tf.exp(reg_cost)

        # index
        d_idx = tf.cast(depth_index, tf.float32)
        if inverse_depth:
            inv_depth_start = tf.div(1.0, depth_start)
            inv_depth_end = tf.div(1.0, depth_end)
            inv_interval = (inv_depth_start - inv_depth_end) / (tf.cast(depth_num, 'float32') - 1)
            inv_depth = inv_depth_start - d_idx * inv_interval
            depth = tf.div(1.0, inv_depth)
        else:
            depth = depth_start + d_idx * depth_interval
        temp_depth_image = tf.reshape(depth, [FLAGS.batch_size, 1, 1, 1])
        temp_depth_image = tf.tile(
            temp_depth_image, [1, feature_shape1[1] * 2, feature_shape1[2] * 2, 1])

        # update the best
        update_flag_image = tf.cast(tf.less(max_prob_image, prob), dtype='float32')
        new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image
        new_depth_image = update_flag_image * temp_depth_image + (1 - update_flag_image) * depth_image
        max_prob_image = tf.assign(max_prob_image, new_max_prob_image)
        depth_image = tf.assign(depth_image, new_depth_image)

        # update counter
        exp_sum = tf.assign_add(exp_sum, prob)
        depth_index = tf.add(depth_index, incre)

        return depth_index, state1, state2, state3, state4, depth_image, max_prob_image, exp_sum, incre

    # run forward loop
    exp_sum = tf.assign(exp_sum, init_map)
    depth_image = tf.assign(depth_image, init_map)
    max_prob_image = tf.assign(max_prob_image, init_map)
    depth_index = tf.constant(0)
    incre = tf.constant(1)
    cond = lambda depth_index, *_: tf.less(depth_index, depth_num)
    _, state1, state2, state3, state4, depth_image, max_prob_image, exp_sum, incre = tf.while_loop(
        cond, body
        , [depth_index, state1, state2, state3, state4, depth_image, max_prob_image, exp_sum, incre]
        , back_prop=False, parallel_iterations=1)

    # get output
    forward_exp_sum = exp_sum + 1e-7
    forward_depth_map = depth_image

    return forward_depth_map, max_prob_image / forward_exp_sum



# loss
def tr_non_zero_mean_absolute_diff(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch """
    with tf.name_scope('MAE'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
        masked_abs_error = tf.abs(mask_true * (y_true - y_pred))  # 4D
        masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])  # 1D
        masked_mae = tf.reduce_sum((masked_mae) / denom)  # 1

    return masked_mae


def tr_less_three_percentage(y_true, y_pred, interval):
    """ less three interval accuracy for one batch """
    with tf.name_scope('less_three_error'):
        shape = tf.shape(y_pred)
        mask_true2 = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true2) + 1e-7
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_three_image = mask_true2 * tf.cast(tf.less_equal(abs_diff_image, 3), dtype='float32')

    return tf.reduce_sum(less_three_image) / denom


def tr_less_one_percentage(y_true, y_pred, interval):
    """ less one interval accuracy for one batch """
    with tf.name_scope('less_one_error'):
        mask_true2 = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.reduce_sum(mask_true2) + 1e-7
        abs_diff_image = tf.abs(y_true - y_pred)
        less_one_image = mask_true2 * tf.cast(tf.less_equal(abs_diff_image, 1.0), dtype='float32')

    return tf.reduce_sum(less_one_image) / denom



def non_zero_mean_absolute_diff(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch """
    with tf.name_scope('MAE'):
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        less_masked_mae = tf.cast(tf.less_equal(tf.abs((y_true - y_pred)) , 10), dtype='float32')
        mask_true2=tf.abs(mask_true * less_masked_mae)
        denom = tf.reduce_sum(mask_true2, axis=[1, 2, 3]) + 1e-7

        masked_abs_error = tf.abs(mask_true2 * (y_true - y_pred))            # 4D
        masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])        # 1D
        masked_mae = tf.reduce_sum(masked_mae / denom)         # 1

    return masked_mae

def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    with tf.name_scope('less_one_error'):
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        less_masked_mae = tf.cast(tf.less_equal(tf.abs((y_true - y_pred)), 1000), dtype='float32')
        mask_true2 = tf.abs(mask_true * less_masked_mae)

        denom = tf.reduce_sum(mask_true2) + 1e-7
        abs_diff_image = tf.abs(y_true - y_pred)
        less_one_image = mask_true2 * tf.cast(tf.less_equal(abs_diff_image, 1.0), dtype='float32')

    return tf.reduce_sum(less_one_image) / denom

def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    with tf.name_scope('less_three_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        less_masked_mae = tf.cast(tf.less_equal(tf.abs((y_true - y_pred)) , 1000), dtype='float32')
        mask_true2=tf.abs(mask_true * less_masked_mae)

        denom = tf.reduce_sum(mask_true2) + 1e-7
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_three_image = mask_true * tf.cast(tf.less_equal(abs_diff_image, 3.0), dtype='float32')

    return tf.reduce_sum(less_three_image) / denom

def less_zerosix_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    with tf.name_scope('less_six_error'):
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        less_masked_mae = tf.cast(tf.less_equal(tf.abs((y_true - y_pred)) , 1000), dtype='float32')
        mask_true2 = tf.abs(mask_true * less_masked_mae)

        denom = tf.reduce_sum(mask_true2) + 1e-7
        abs_diff_image = tf.abs(y_true - y_pred)
        less_one_image = mask_true2 * tf.cast(tf.less_equal(abs_diff_image, 0.6), dtype='float32')

    return tf.reduce_sum(less_one_image) / denom


def test_depth_loss(estimated_depth_image, depth_image, depth_interval):
    """ compute loss and accuracy """
    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(depth_image, estimated_depth_image, depth_interval)
    # less one accuracy
    less_one_accuracy = less_one_percentage(depth_image, estimated_depth_image, depth_interval)
    # less three accuracy
    less_three_accuracy = less_three_percentage(depth_image, estimated_depth_image, depth_interval)
    # less 0.6 accuracy
    less_six_accuracy = less_zerosix_percentage(depth_image, estimated_depth_image, depth_interval)

    return masked_mae, less_one_accuracy, less_three_accuracy, less_six_accuracy


def tr_classification_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
    """ compute loss and accuracy """

    image_shape = tf.shape(gt_depth_image)
    gt_depth_image = tf.image.resize_bilinear(gt_depth_image, [image_shape[1], image_shape[2]])

    # get depth mask
    mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    valid_pixel_num = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    # gt depth map -> gt index map
    shape = tf.shape(gt_depth_image)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    start_mat = tf.tile(tf.reshape(depth_start, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])

    interval_mat = tf.tile(tf.reshape(depth_interval, [shape[0], 1, 1, 1]), [1, shape[1], shape[2], 1])
    gt_index_image = tf.div(gt_depth_image - start_mat, interval_mat)
    gt_index_image = tf.multiply(mask_true, gt_index_image)
    gt_index_image = tf.cast(tf.round(gt_index_image), dtype='int32')
    # gt index map -> gt one hot volume (B x H x W x 1)
    gt_index_volume = tf.one_hot(gt_index_image, depth_num, axis=1)
    # cross entropy image (B x H x W x 1)
    cross_entropy_image = -tf.reduce_sum(gt_index_volume * tf.log(tf.clip_by_value(prob_volume, 1e-10, 1.0)), axis=1)
    # masked cross entropy loss
    masked_cross_entropy_image = tf.multiply(mask_true, cross_entropy_image)
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy_image, axis=[1, 2, 3])
    masked_cross_entropy = tf.reduce_sum(masked_cross_entropy / valid_pixel_num)

    # winner-take-all depth map
    wta_index_map = tf.cast(tf.argmax(prob_volume, axis=1), dtype='float32')

    wta_depth_map = wta_index_map * interval_mat + start_mat

    # non zero mean absulote loss
    masked_mae = tr_non_zero_mean_absolute_diff(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less one accuracy
    less_one_accuracy = tr_less_one_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less three accuracy
    less_three_accuracy = tr_less_three_percentage(gt_depth_image, wta_depth_map, tf.abs(depth_interval))

    return masked_cross_entropy, masked_mae, less_one_accuracy, less_three_accuracy, wta_depth_map