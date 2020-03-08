#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Training script.
"""

from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import time
import sys
import math
import argparse
from random import randint
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tfdbg
from PIL import Image
#from scipy.misc import imread
from imageio import imread
import matplotlib.pyplot as plt

from preprocess import *
from model import *

# paths
tf.app.flags.DEFINE_string('data_root', 'X:/liujin_densematching/MVS_traindata/meitan_RS/train', """Path to whu train dataset.""")

tf.app.flags.DEFINE_string('log_dir', 'MVS_TRANING/tf_log',
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', 'MVS_TRANING/tf_model',
                           """Path to save the model.""")
tf.app.flags.DEFINE_boolean('use_pretrain', False,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step', 110000,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 3, 
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 768, 
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 384, 
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.5,
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('interval_scale', 2,
                            """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('interval', 0.1, 
                            """Depth interval for building cost volume.""")

# training parameters
tf.app.flags.DEFINE_integer('num_gpus', 1, 
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', 21, 
                            """Training epoch number.""")
tf.app.flags.DEFINE_float('val_ratio', 0, 
                          """Ratio of validation set when splitting dataset.""")
tf.app.flags.DEFINE_float('base_lr', 0.001,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 5000,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000,
                            """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.9,
                          """Learning rate decay rate.""")

FLAGS = tf.app.flags.FLAGS

class MVSGenerator:
    """ data generator class, tf only accept generator without param """
    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0
    
    def __iter__(self):
        while True:
            for data in self.sample_list: 
                start_time = time.time()

                ###### read input data ######
                images = []
                cams = []
                for view in range(self.view_num):
                    image = image_augment(Image.open(data[2 * view]))
                    image = center_image(image)
                    cam = tr_load_cam(open(data[2 * view + 1]), FLAGS.interval_scale)
                    images.append(image)
                    cams.append(cam)

                depimg = imread(os.path.join(data[2 * self.view_num]))
                depth_image = (np.float32(depimg) / 64.0)   # WHU MVS dataset

                scaled_cams = scale_mvs_camera(cams, scale=FLAGS.sample_scale)

                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]#
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # return mvs input
                self.counter += 1
                duration = time.time() - start_time
                images = np.stack(images, axis=0)
                scaled_cams = np.stack(scaled_cams, axis=0)

                yield (images, scaled_cams, cams, depth_image)

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(traning_list):
    """ training rednet """
    training_sample_size = len(traning_list)
    print ('sample number: ', training_sample_size)

    with tf.Graph().as_default(), tf.device('/cpu:0'): 

        ########## data iterator #########
        # training generators
        training_generator = iter(MVSGenerator(traning_list, FLAGS.view_num))
        generator_data_type = (tf.float32, tf.float32, tf.float32, tf.float32)
        # dataset from generator
        training_set = tf.data.Dataset.from_generator(lambda: training_generator, generator_data_type)
        training_set = training_set.batch(FLAGS.batch_size)
        training_set = training_set.prefetch(buffer_size=1)
        # iterators
        training_iterator = training_set.make_initializable_iterator()

        ########## optimization options ##########
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr_op = tf.train.exponential_decay(FLAGS.base_lr, global_step=global_step, 
                                           decay_steps=FLAGS.stepvalue, decay_rate=FLAGS.gamma, name='lr')
        opt = tf.train.RMSPropOptimizer(learning_rate=lr_op)

        tower_grads = []
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data
                    images, scale_cams, cams, depth_image = training_iterator.get_next()
                    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
                    scale_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
                    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
                    depth_image.set_shape(tf.TensorShape([None, None, None, 1]))
                    depth_start = tf.reshape(
                        tf.slice(scale_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(
                        tf.slice(scale_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])

                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True

                    ## inference
                    # probability volume
                    prob_volume = inference_prob_recurrent(
                        images, scale_cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                    # classification loss
                    loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                        tr_classification_loss(
                            prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)

                    # retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # keep track of the gradients across all towers.
                    tower_grads.append(grads)
        
        # average gradient
        grads = average_gradients(tower_grads)
        
        # training opt
        train_opt = opt.apply_gradients(grads, global_step=global_step)

        # summary 
        summaries.append(tf.summary.scalar('loss', loss))
        summaries.append(tf.summary.scalar('less_one_meter_accuracy', less_one_accuracy))
        summaries.append(tf.summary.scalar('less_three_interval_accuracy', less_three_accuracy))
        summaries.append(tf.summary.scalar('lr', lr_op))
        weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in weights_list:
            summaries.append(tf.summary.histogram(var.op.name, var))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        
        # saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)        
        summary_op = tf.summary.merge(summaries)

        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:     
            
            # initialization
            total_step = 0
            sess.run(init_op)
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

            # load pre-trained model
            if FLAGS.use_pretrain:
                pretrained_model_path = os.path.join(FLAGS.model_dir, 'model.ckpt')
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(sess, '-'.join([pretrained_model_path, str(FLAGS.ckpt_step)]))
                print('Pre-trained model restored from %s' %
                    ('-'.join([pretrained_model_path, str(FLAGS.ckpt_step)])))
                total_step = FLAGS.ckpt_step

            # training several epochs
            for epoch in range(FLAGS.epoch):

                # training of one epoch
                step = 0
                sess.run(training_iterator.initializer)
                for _ in range(int(training_sample_size / FLAGS.num_gpus)):

                    # run one batch
                    start_time = time.time()
                    try:
                        out_summary_op, out_opt, out_loss, out_less_one, out_less_three = sess.run(
                        [summary_op, train_opt, loss, less_one_accuracy, less_three_accuracy])
                    except tf.errors.OutOfRangeError:
                        print("End of dataset")  # ==> "End of dataset"
                        break
                    duration = time.time() - start_time

                    # print info
                    if step % FLAGS.display == 0:
                        print('epoch, %d, step %d, total_step %d, loss = %.4f, (< 1m) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                            (epoch, step, total_step, out_loss, out_less_one, out_less_three, duration))
                    
                    # write summary
                    if step % (FLAGS.display * 10) == 0:
                        summary_writer.add_summary(out_summary_op, total_step)
                   
                    # save the model checkpoint periodically
                    if (total_step % FLAGS.snapshot == 0 or step == (training_sample_size - 1)):
                        model_folder = os.path.join(FLAGS.model_dir)
                        if not os.path.exists(model_folder):
                            os.mkdir(model_folder)
                        ckpt_path = os.path.join(model_folder, 'model.ckpt')
                        print('Saving model to %s' % ckpt_path)
                        saver.save(sess, ckpt_path, global_step=total_step)
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus

def main(argv=None):
    """ program entrance """

    # Prepare all training samples
    sample_list = gen_train_mvs_list(FLAGS.data_root)

    # Shuffle
    random.shuffle(sample_list)

    # Training entrance.
    train(sample_list)


if __name__ == '__main__':

    print ('Training RED-Net with %d views' % FLAGS.view_num)

    tf.app.run()