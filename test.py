#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Test script.
"""

from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import sys
import math
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
#from scipy.misc import imread
from imageio import imread, imwrite, imsave

from preprocess import *
from model import *

# dataset parameters
tf.app.flags.DEFINE_string('dense_folder', 'X:/liujin_densematching/MVS_traindata/meitan_RS/test/',
                           """Root path to dense folder.""")
tf.app.flags.DEFINE_string('model_dir', 
                           'MODEL_FOLDER',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 150000,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 3,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 200,
                            """Maximum depth step when testing.""")
tf.app.flags.DEFINE_integer('max_w', 768,
                            """Maximum image width when testing.""")
tf.app.flags.DEFINE_integer('max_h', 384,
                            """Maximum image height when testing.""")
tf.app.flags.DEFINE_float('resize_scale', 1,
                            """output scale for depth and image (W and H).""")
tf.app.flags.DEFINE_float('sample_scale', 0.5, 
                            """Downsample scale for building cost volume (W and H).""")
tf.app.flags.DEFINE_float('interval_scale', 1,
                            """Downsample scale for building cost volume (D).""")
tf.app.flags.DEFINE_float('base_image_size', 8, 
                            """Base image size""")
tf.app.flags.DEFINE_integer('batch_size', 1, 
                            """Testing batch size.""")
tf.app.flags.DEFINE_bool('adaptive_scaling', True, 
                            """Let image size to fit the network, including 'scaling', 'cropping'""")

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
                
                # read input data
                images = []
                cams = []
                locations = []
                selected_view_num = int(len(data) / 2)

                for view in range(min(self.view_num, selected_view_num)):
                    image_file = file_io.FileIO(data[2 * view], mode='rb')
                    image = imread(image_file)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cam_file = file_io.FileIO(data[2 * view + 1], mode='rb')
                    cam, location = load_cam_whutest(cam_file, FLAGS.interval_scale)
                    location.append(str(FLAGS.resize_scale))

                    if cam[1][3][2] == 0:
                        cam[1][3][2] = FLAGS.max_d
                    start = cam[1][3][0]
                    interval = cam[1][3][1]
                    end = cam[1][3][3]
                    maxd = cam[1][3][2]
                    
                    images.append(image)
                    cams.append(cam)
                    locations.append(location)

                if selected_view_num < self.view_num:
                    for view in range(selected_view_num, self.view_num):
                        image_file = file_io.FileIO(data[0], mode='rb')
                        image = imread(image_file)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        cam_file = file_io.FileIO(data[1], mode='rb')
                        cam = load_cam_whutest(cam_file, FLAGS.interval_scale)
                        cam[1][3][0] = start
                        cam[1][3][1] = interval
                        cam[1][3][2] = maxd
                        cam[1][3][3] = end

                        images.append(image)
                        cams.append(cam)

                depimg = imread(os.path.join(data[2 * self.view_num]))
                depth_image = (np.float32(depimg)/64.0)   # whu dataset

                # mask out-of-range depth pixels (in a relaxed range)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)

                # determine a proper scale to resize input 
                resize_scale = 1
                if FLAGS.adaptive_scaling:
                    h_scale = 0
                    w_scale = 0
                    for view in range(self.view_num):
                        height_scale = float(FLAGS.max_h) / images[view].shape[0]
                        width_scale = float(FLAGS.max_w) / images[view].shape[1]
                        if height_scale > h_scale:
                            h_scale = height_scale
                        if width_scale > w_scale:
                            w_scale = width_scale
                    if h_scale > 1 or w_scale > 1:
                        print ("max_h, max_w should < W and H!")
                        exit(-1)
                    resize_scale = h_scale
                    if w_scale > h_scale:
                        resize_scale = w_scale
                scaled_input_images, scaled_input_cams = scale_mvs_input(images, cams, scale=resize_scale)
                locations[0][-1] = str(float(locations[0][-1])*resize_scale)

                # crop to fit network
                croped_images, croped_cams = crop_mvs_input(scaled_input_images, scaled_input_cams)

                # center images
                centered_images = []
                for view in range(self.view_num):
                    centered_images.append(center_image(croped_images[view]))

                # sample cameras for building cost volume
                real_cams = np.copy(croped_cams) 
                scaled_cams = scale_mvs_camera(croped_cams, scale=FLAGS.sample_scale)

                # return mvs input
                scaled_images = []
                for view in range(self.view_num):
                    scaled_images.append(scale_image(croped_images[view], scale=FLAGS.resize_scale))
                    
                scaled_images = np.stack(scaled_images, axis=0)
                croped_images = np.stack(croped_images, axis=0)
                scaled_cams = np.stack(scaled_cams, axis=0)
                self.counter += 1
                yield (scaled_images, centered_images, scaled_cams, real_cams, locations, depth_image) 

def rednet_pipeline(mvs_list):

    """ rednet in altizure pipeline """
    print('sample number: ', len(mvs_list))

    # create output folder
    output_folder = os.path.join(FLAGS.dense_folder, 'depths_rednet')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # testing set
    mvs_generator = iter(MVSGenerator(mvs_list, FLAGS.view_num))
    generator_data_type = (tf.float32, tf.float32, tf.float32, tf.float32, tf.string, tf.float32)   
    mvs_set = tf.data.Dataset.from_generator(lambda: mvs_generator, generator_data_type)
    mvs_set = mvs_set.batch(FLAGS.batch_size)
    mvs_set = mvs_set.prefetch(buffer_size=1)

    # data from dataset via iterator
    mvs_iterator = mvs_set.make_initializable_iterator()
    scaled_images, centered_images, scaled_cams, croped_cams, locations, depth_image = mvs_iterator.get_next()

    # set shapes
    scaled_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    centered_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    scaled_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    croped_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_image.set_shape(tf.TensorShape([None, None, None, 1]))

    depth_start = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_interval = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_num = tf.cast(
        tf.reshape(tf.slice(scaled_cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), 'int32')

    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # depth map inference using red
    init_depth_map, prob_map = inference_winner_take_all(centered_images, scaled_cams, depth_num, depth_start, depth_end)

    # test loss
    image_shape = tf.shape(depth_image)
    upsample_depth_map = tf.image.resize_bilinear(init_depth_map, [image_shape[1], image_shape[2]])
    loss, less_one_temp, less_three_temp, less_zerosix_temp = test_depth_loss(upsample_depth_map, depth_image, depth_interval)


    # init option
    init_op = tf.global_variables_initializer()
    var_init_op = tf.local_variables_initializer()

    # GPU grows incrementally
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:   

        # initialization
        sess.run(var_init_op)
        sess.run(init_op)
        total_step = 0
        total_loss = 0
        total_less_one = 0
        total_less_three = 0
        total_less_six = 0

        # load model
        if FLAGS.model_dir is not None:
            pretrained_model_ckpt_path = os.path.join(FLAGS.model_dir, 'model.ckpt')
            restorer = tf.train.Saver(tf.global_variables())
            restorer.restore(sess, '-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
            print('Pre-trained model restored from %s' %
                  ('-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])))
            total_step = FLAGS.ckpt_step
    
        # run inference for each reference view
        sess.run(mvs_iterator.initializer)
        for step in range(len(mvs_list)):

            start_time = time.time()
            try:
                out_init_depth_map, out_prob_map, out_images, out_cams, out_croped_cams, out_locations, out_loss, out_less_one, out_less_three, out_less_zerosix  = sess.run(
                    [init_depth_map, prob_map, scaled_images, scaled_cams, croped_cams, locations, loss, less_one_temp, less_three_temp, less_zerosix_temp])
            except tf.errors.OutOfRangeError:
                print("all dense finished")  # ==> "End of dataset"
                break
            duration = time.time() - start_time
            print('depth inference %d finished, loss = %.4f, (< 1m) = %.4f, (< 3px) = %.4f , (< 0.6m) = %.4f , (%.3f sec/step)' %(step, out_loss, out_less_one, out_less_three, out_less_zerosix, duration))

            # squeeze output
            out_init_depth_image = np.squeeze(out_init_depth_map)
            out_prob_map = np.squeeze(out_prob_map)
            out_prob_map[np.isnan(out_prob_map)] = 1e-10
            out_ref_image = np.squeeze(out_images)
            out_ref_image = np.squeeze(out_ref_image[0, :, :, :])
            out_ref_cam = np.squeeze(out_croped_cams)
            out_ref_cam = np.squeeze(out_ref_cam[0, :, :, :])

            out_location = np.squeeze(out_locations)
            out_location = np.squeeze(out_location[0, :])
            out_index = out_location[0].decode('utf-8')
            rowindex = int(out_location[3])
            colindex = int(out_location[4])

            # paths
            output_folder2 = output_folder + ('/%s/' % out_index)
            if not os.path.exists(output_folder2):
            	os.mkdir(output_folder2)
            init_depth_map_path = output_folder2 + ('%03d%03d_init.pfm' % (rowindex, colindex))
            prob_map_path = output_folder2 + ('%03d%03d_prob.pfm' % (rowindex, colindex))
            out_ref_image_path = output_folder2 + ('%03d%03d.jpg' % (rowindex, colindex))
            out_ref_cam_path = output_folder2 + ('%03d%03d.txt' % (rowindex, colindex))

            # save output
            write_pfm(init_depth_map_path, out_init_depth_image)
            write_pfm(prob_map_path, out_prob_map)
            out_ref_image = cv2.cvtColor(out_ref_image, cv2.COLOR_RGB2BGR)
            image_file = file_io.FileIO(out_ref_image_path, mode='w')
            imwrite(out_ref_image_path, np.uint8(out_ref_image))
            write_cam(out_ref_cam_path, out_ref_cam, out_location)
            total_loss += np.squeeze(out_loss)
            total_less_one += np.squeeze(out_less_one)
            total_less_three += np.squeeze(out_less_three)
            total_less_six += np.squeeze(out_less_zerosix)
            total_step = step + 1

        total_loss = total_loss/total_step
        total_less_one = total_less_one/total_step
        total_less_three = total_less_three/total_step
        total_less_six = total_less_six/total_step
        print('total %d finished, total_loss = %.4f, (total < 1m) = %.4f, (total < 3px) = %.4f, (total < 0.6m) = %.4f' %
                (total_step, total_loss, total_less_one, total_less_three, total_less_six))


def main(_):
    """ program entrance """
    # generate input path list
    mvs_list = gen_test_mvs_list(FLAGS.dense_folder)

    # rednet inference
    rednet_pipeline(mvs_list)


if __name__ == '__main__':

    tf.app.run()