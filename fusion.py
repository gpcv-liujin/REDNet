#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Fuse rednet results.
"""
import argparse
import os
import numpy as np
import time
import re
import sys
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import threading

parser = argparse.ArgumentParser(description='filter and fuse.')
parser.add_argument('--dense_folder', type=str,
                    default='X:/liujin_densematching/MVS_traindata/meitan_RS/test_largeimage/')
parser.add_argument('--confidence_ratio', default=0.2, help='confidence_ratio')
parser.add_argument('--geo_consist_num', default=3, help='geo_consist_num')
parser.add_argument('--skip_line', default=1, help='skip_line')
parser.add_argument('--camera_scale', default=1, help='camera_scale')
parser.add_argument('--Negative_depth', default=True, help='Negative_depth')  # Negative depth for XrightYup

# this cord input Twc [Rwc| twc], and K[3*3],
# if camera orientation is XrightYup, Negative_depth==True; else if camera orientation is XrightYdown, Negative_depth==False;
# keep Negative_depth==True for rednet results.

args = parser.parse_args()

# read intrinsics and extrinsics
def read_camera_parameters(filename, scale=1):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix  Twc
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    extrinsics = np.linalg.inv(extrinsics)
    # intrinsics: line [7-10), 3x3 matrix  K
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is downsampled
    intrinsics[:2, :] *= scale
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        print(num_viewpoint)
        # viewpoints
        for view_idx in range(num_viewpoint):
            views = [ x for x in f.readline().rstrip().split()]
            ref_view = views[0]
            src_views = views[2::2]
            data.append((ref_view, src_views))
    return data

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),   # extrinsics_ref : Tcw
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]

    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(pairlist, scan_folder, out_folder, confidence_ratio, geo_consist_num, skip_line, camera_scale):
    # the pair file
    pair_data = read_pair_file(pairlist)
    nviews = len(pair_data)
    print(nviews)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        if os.path.exists(os.path.join(scan_folder, '{}_init.pfm'.format(ref_view))):
            # for the final point cloud
            plyfilename = os.path.join(out_folder, '{}.ply'.format(ref_view))
            vertexs = []
            vertex_colors = []
            # load the camera parameters
            ref_intrinsics, ref_extrinsics = read_camera_parameters(os.path.join(scan_folder, '{}.txt'.format(ref_view)), camera_scale)
            # load the reference image
            ref_img = read_img(os.path.join(scan_folder, '{}.jpg'.format(ref_view)))
            # load the estimated depth of the reference view
            ref_depth_est = read_pfm(os.path.join(scan_folder, '{}_init.pfm'.format(ref_view)))[0]
            if args.Negative_depth == True:
                ref_depth_est = - ref_depth_est
            # load the photometric mask of the reference view
            confidence = read_pfm(os.path.join(scan_folder, '{}_prob.pfm'.format(ref_view)))[0]
            photo_mask = confidence > confidence_ratio

            all_srcview_depth_ests = []
            all_srcview_x = []
            all_srcview_y = []
            all_srcview_geomask = []

            # compute the geometric mask
            geo_mask_sum = 0
            for src_view in src_views:
                if os.path.exists(os.path.join(scan_folder, '{}_init.pfm'.format(src_view))):
                    # camera parameters of the source view
                    src_intrinsics, src_extrinsics = read_camera_parameters(os.path.join(scan_folder, '{}.txt'.format(src_view)),camera_scale)
                    # the estimated depth of the source view
                    src_depth_est = read_pfm(os.path.join(scan_folder, '{}_init.pfm'.format(src_view)))[0]
                    if args.Negative_depth:
                        src_depth_est = - src_depth_est
                    geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                              src_depth_est,
                                                                              src_intrinsics, src_extrinsics)
                    geo_mask_sum += geo_mask.astype(np.int32)
                    all_srcview_depth_ests.append(depth_reprojected)
                    all_srcview_x.append(x2d_src)
                    all_srcview_y.append(y2d_src)
                    all_srcview_geomask.append(geo_mask)

            depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
            # at least N source views matched
            geo_mask = geo_mask_sum >= geo_consist_num
            final_mask = np.logical_and(photo_mask, geo_mask)

            os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
            save_mask(os.path.join(out_folder, "mask/{}_photo.png".format(ref_view)), photo_mask)
            save_mask(os.path.join(out_folder, "mask/{}_geo.png".format(ref_view)), geo_mask)
            save_mask(os.path.join(out_folder, "mask/{}_final.png".format(ref_view)), final_mask)
            print("ref-view {}, photo/geo/final-mask:{}/{}/{}".format(ref_view, photo_mask.mean(), geo_mask.mean(), final_mask.mean()))


            height, width = depth_est_averaged.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
            # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
            valid_points = final_mask
            print("valid_points", valid_points.mean())
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
            color = ref_img[valid_points]
            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]
            vertexs.append(xyz_world.transpose((1, 0)))
            vertex_colors.append((color * 255).astype(np.uint8))

            vertexs2 = np.concatenate(vertexs, axis=0)
            vertex_colors2 = np.concatenate(vertex_colors, axis=0)
            vertexs2 = np.array([tuple(v) for v in vertexs2[1::int(skip_line)]], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            vertex_colors2 = np.array([tuple(v) for v in vertex_colors2[1::int(skip_line)]], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

            vertex_all = np.empty(len(vertexs2), vertexs2.dtype.descr + vertex_colors2.dtype.descr)
            for prop in vertexs2.dtype.names:
                vertex_all[prop] = vertexs2[prop]
            for prop in vertex_colors2.dtype.names:
                vertex_all[prop] = vertex_colors2[prop]

            el = PlyElement.describe(vertex_all, 'vertex')
            PlyData([el]).write(plyfilename)
            print("saving the final model to", plyfilename)


if __name__ == '__main__':

    pairlist = args.dense_folder + '/viewpair.txt'
    testpath = os.path.join(args.dense_folder, 'depths_rednet')
    outdir = os.path.join(args.dense_folder, 'rednet_fusion')

    filter_depth(pairlist, testpath, outdir, args.confidence_ratio, args.geo_consist_num, args.skip_line,
                 args.camera_scale)
