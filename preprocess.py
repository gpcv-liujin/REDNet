#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
data preprocesses.
"""

from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import time
import glob
import random
import math
import re
import sys

import cv2
import numpy as np
import random
import tensorflow as tf
import scipy.io
import urllib
from tensorflow.python.lib.io import file_io
from PIL import Image, ImageEnhance, ImageOps, ImageFile
FLAGS = tf.app.flags.FLAGS


def center_image(img):
    """ normalize image input """
    img_array = np.array(img)
    img = img_array.astype(np.float32)
    #img = img.astype(np.float32)
    var = np.var(img, axis=(0,1), keepdims=True)
    mean = np.mean(img, axis=(0,1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)

def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal: 
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale

    return new_cam

def scale_mvs_camera(cams, scale=1):
    """ resize input in order to produce sampled depth map """
    for view in range(FLAGS.view_num):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams

def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def scale_mvs_input(images, cams, depth_image=None, scale=1):
    """ resize input to fit into the memory """
    for view in range(FLAGS.view_num):
        images[view] = scale_image(images[view], scale=scale)
        cams[view] = scale_camera(cams[view], scale=scale)

    if depth_image is None:
        return images, cams
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='nearest')
        return images, cams, depth_image

def crop_mvs_input(images, cams, depth_image=None):
    """ resize images and cameras to fit the network (can be divided by base image size) """

    # crop images and cameras
    max_h = int(FLAGS.max_h * FLAGS.resize_scale)
    max_w = int(FLAGS.max_w * FLAGS.resize_scale)
    for view in range(FLAGS.view_num):
        h, w = images[view].shape[0:2]
        new_h = h
        new_w = w
        if new_h > max_h:
            new_h = max_h
        else:
            new_h = int(math.ceil(h / FLAGS.base_image_size) * FLAGS.base_image_size)
        if new_w > max_w:
            new_w = max_w
        else:
            new_w = int(math.ceil(w / FLAGS.base_image_size) * FLAGS.base_image_size)
        start_h = int(math.ceil((h - new_h) / 2))
        start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        images[view] = images[view][start_h:finish_h, start_w:finish_w]
        cams[view][1][0][2] = cams[view][1][0][2] - start_w
        cams[view][1][1][2] = cams[view][1][1][2] - start_h

    # crop depth image
    if not depth_image is None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return images, cams, depth_image
    else:
        return images, cams

def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image


def tr_load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    pera = np.zeros((1, 13))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]  # Rwc

    for i in range(0, 13):
        pera[0][i] = words[17 + i]

    f = pera[0][0]
    x0 = pera[0][1]  # whu
    y0 = pera[0][2]

    # K
    cam[1][0][0] = -f
    cam[1][1][1] = f
    cam[1][0][2] = x0
    cam[1][1][2] = y0
    cam[1][2][2] = 1

    # depth range
    cam[1][3][0] = pera[0][3]  # start
    cam[1][3][1] = pera[0][5] * interval_scale  # interval
    cam[1][3][3] = pera[0][4]  # end

    acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 8 + 1) * 8

    if acturald > FLAGS.max_d:
        scale = acturald / float(FLAGS.max_d)
        cam[1][3][1] = cam[1][3][1] * scale
        acturald = FLAGS.max_d

    cam[1][3][2] = acturald

    return cam


def load_cam_whutest(file, interval_scale=1):
    """ read camera txt file (XrightYup，Twc)"""
    cam = np.zeros((2, 4, 4))
    pera = np.zeros((1, 13))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]  # Rwc

    for i in range(0, 13):
        pera[0][i] = words[17 + i]

    f = pera[0][0]
    x0 = pera[0][1]      # WHU test set
    y0 = pera[0][2]

    # K   XrightYup
    cam[1][0][0] = -f
    cam[1][1][1] = f
    cam[1][0][2] = x0
    cam[1][1][2] = y0
    cam[1][2][2] = 1

    # depth range
    cam[1][3][0] = pera[0][3]  # start
    cam[1][3][1] = pera[0][5] * interval_scale  # interval
    cam[1][3][3] = pera[0][4]  # end

    acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 8 + 1) * 8

    if acturald > FLAGS.max_d:
        scale = acturald / float(FLAGS.max_d)
        cam[1][3][1] = cam[1][3][1] * scale
        acturald = FLAGS.max_d  # maxd

    cam[1][3][2] = acturald  # maxd

    location = words[23:30]

    return cam, location


def load_cam_truemeitan(file, interval_scale=1):
    """ read camera txt file  (XrightYdown，[Rcw|twc])"""
    cam = np.zeros((2, 4, 4))
    pera=np.zeros((1, 13))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]  # Rcw

    for i in range(0, 13):
        pera[0][i] = words[17+i]

    f  = pera[0][0]
    x0 = pera[0][1]
    y0 = pera[0][2]

    # trans Rcw to Rwc
    R = cam[0, 0:3, 0:3]
    cam[0, 0:3, 0:3] = np.linalg.inv(R)

    # K   XrightYdown
    cam[1][0][0] = f
    cam[1][1][1] = f
    cam[1][0][2] = x0
    cam[1][1][2] = y0
    cam[1][2][2] = 1
    cam[0][0][3] = - cam[0][0][3]
    cam[0][1][3] = - cam[0][1][3]
    cam[0][2][3] = - cam[0][2][3]

    cam[1][3][0] = pera[0][3]  #start
    cam[1][3][1] = pera[0][5] * interval_scale   #interval
    cam[1][3][3] = pera[0][4]   #end

    acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 8 + 1) * 8

    if acturald > FLAGS.max_d:
        scale = acturald / float(FLAGS.max_d)
        cam[1][3][1] = cam[1][3][1] * scale
        acturald = FLAGS.max_d

    cam[1][3][2] = acturald
    location = words[23:30]

    return cam, location


def write_cam(file, cam, location):
    # f = open(file, "w")
    f = file_io.FileIO(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    for word in location:
        f.write(str(word.decode('utf-8')) + ' ')
    f.write('\n')

    f.close()

def load_pfm(fname):
    
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    file = open(fname,'rb')
    header = str(file.readline().decode('latin-1')).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline().decode('latin-1')).rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian


    data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data


def write_pfm(file, image, scale=1):
    file = file_io.FileIO(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image_string = image.tostring()
    file.write(image_string)    

    file.close()


# For training
def gen_train_mvs_list(data_folder, mode='training'):
    """ generate data paths for whu dataset """
    sample_list = []
    
    # parse camera pairs
    cluster_file_path = data_folder + '/pair.txt'

    cluster_list = file_io.FileIO(cluster_file_path, mode='r').read().split()

    # 3 sets
    train_cluster_path = data_folder + '/index.txt'
    training_set = file_io.FileIO(train_cluster_path, mode='r').read().split()


    data_set = []
    if mode == 'training':
        data_set = training_set

    # for each dataset
    for i in data_set:
        image_folder = os.path.join(data_folder, ('Images/%s' % i))
        cam_folder = os.path.join(data_folder, ('Cams/%s' % i))
        depth_folder = os.path.join(data_folder, ('Depths/%s' % i))

        if mode == 'training':
            # for each view
            for p in range(0, int(cluster_list[0])): # 0-4
                index_ref = int(cluster_list[(int(cluster_list[0])+1) * p + 1])
                image_folder2 = os.path.join(image_folder, ('%d' % index_ref))
                image_files = sorted(os.listdir(image_folder2))

                for j in range(0,int(np.size(image_files))):
                    paths = []
                    portion = os.path.splitext(image_files[j])
                    newcamname = portion[0] + '.txt'
                    newdepthname = portion[0] + '.png'

                    # ref image
                    ref_image_path = os.path.join(os.path.join(image_folder, ('%d' % index_ref)), image_files[j])
                    ref_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % index_ref)), newcamname)
                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)

                    # view images
                    for view in range(FLAGS.view_num - 1):
                        index_view = int(cluster_list[(int(cluster_list[0])+1) * p + 3 + view])  # selected view image
                        view_image_path = os.path.join(os.path.join(image_folder, ('%d' % index_view)), image_files[j])
                        view_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % index_view)), newcamname)
                        paths.append(view_image_path)
                        paths.append(view_cam_path)

                    # depth path
                    depth_image_path = os.path.join(os.path.join(depth_folder, ('%d' % index_ref)), newdepthname)   
                    paths.append(depth_image_path)
                    sample_list.append(paths)

    return sample_list


# for testing
def gen_test_mvs_list(dense_folder):
    """ mvs input path list """

    cluster_list_path = os.path.join(dense_folder, 'pair.txt')
    cluster_list = open(cluster_list_path).read().split()

     # test sets
    test_cluster_path = dense_folder + '/index.txt'
    test_set = file_io.FileIO(test_cluster_path, mode='r').read().split()

    # for each dataset
    mvs_list = []
    for m in test_set:
        image_folder = os.path.join(dense_folder, ('Images/%s' % m))
        cam_folder = os.path.join(dense_folder, ('Cams/%s' % m))
        depth_folder = os.path.join(dense_folder, ('Depths/%s' % m))

        for i in range(int(cluster_list[0])):# 0-4
            index_ref=int(cluster_list[(int(cluster_list[0])+1) * i + 1])
            image_folder2=os.path.join(image_folder, ('%d' % index_ref))
            image_files = sorted(os.listdir(image_folder2))  

            for j in range(0,int(np.size(image_files))):
                paths = []
                portion = os.path.splitext(image_files[j])   
                newcamname = portion[0] + '.txt'
                newdepthname = portion[0] + '.png'
                #newdepthname = portion[0] + '.pfm'

                # ref image
                ref_image_path = os.path.join(os.path.join(image_folder, ('%d' % index_ref)), image_files[j])
                ref_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % index_ref)), newcamname)
                paths.append(ref_image_path)
                paths.append(ref_cam_path)

                # view images
                all_view_num = int(cluster_list[2])
                check_view_num = min(FLAGS.view_num - 1, all_view_num)
                for view in range(check_view_num):
                    index_view = int(cluster_list[(int(cluster_list[0])+1) * i + 3 + view]) # selected view image
                    view_image_path = os.path.join(os.path.join(image_folder, ('%d' % index_view)), image_files[j])
                    view_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % index_view)), newcamname)
                    paths.append(view_image_path)
                    paths.append(view_cam_path)

                # depth path
                ref_depth_path = os.path.join(os.path.join(depth_folder, ('%d' % index_ref)), newdepthname)
                paths.append(ref_depth_path)
                mvs_list.append(paths)

    return mvs_list


# for predict without depth
def gen_predict_mvs_list(dense_folder, view_num):
    """ mvs input path list """

     # 3 sets
    test_cluster_path = dense_folder + '/viewpair.txt'
    cluster_list = file_io.FileIO(test_cluster_path, mode='r').read().split()

    image_folder = os.path.join(dense_folder, 'Images')
    cam_folder = os.path.join(dense_folder, 'Cams')

    # for each dataset
    mvs_list = []
    total_num = int(cluster_list[0])
    all_view_num = int(cluster_list[1])

    for i in range(total_num):# 0-4
        paths = []
        index_ref = cluster_list[(all_view_num) * i * 2 + 2]  # reference
        ref_image_path = os.path.join(image_folder, '{}.png'.format(index_ref))
        ref_cam_path = os.path.join(cam_folder, '{}.txt'.format(index_ref))
        paths.append(ref_image_path)
        paths.append(ref_cam_path)

        # view images
        check_view_num = min(FLAGS.view_num - 1, all_view_num)
        for view in range(check_view_num):
            index_view = cluster_list[(all_view_num) * i * 2 + 4 + view * 2]  # source
            view_image_path = os.path.join(image_folder, '{}.png'.format(index_view))
            view_cam_path = os.path.join(cam_folder, '{}.txt'.format(index_view))
            paths.append(view_image_path)
            paths.append(view_cam_path)

        mvs_list.append(paths)

    return mvs_list


# data augment
def image_augment(image):

    image = randomColor(image)
    #image = randomGaussian(image, mean=0.2, sigma=0.3)

    return image
    

def randomColor(image):

    random_factor = np.random.randint(1, 301) / 100.
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # Image Color
    random_factor = np.random.randint(10, 201) / 100.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # Image Brightness
    random_factor = np.random.randint(10, 201) / 100.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # Image Contrast
    random_factor = np.random.randint(0, 301) / 100.
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # Image Sharpness

    return sharpness_image



def randomGaussian(image, mean=0.02, sigma=0.03):

    def gaussianNoisy(im, mean=0.02, sigma=0.03):


        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    img.flags.writeable = True
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])

    return Image.fromarray(np.uint8(img))
