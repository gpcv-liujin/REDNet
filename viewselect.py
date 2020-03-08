#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
View Selection.
"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='View Selection for dense matching')

parser.add_argument('--dense_folder', type=str,
                    default='X:/liujin_densematching/MVS_traindata/meitan_RS/test_largeimage/')
parser.add_argument('--view_num', type=int, default=20)
parser.add_argument('--Cams_ori', type=int, default=1)  # Camera orientation, 0 XrightYdown; 1 XrightYup

args = parser.parse_args()

# This cord excepts for Twc[Rwc|twc]

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix  Twc
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))   # Twc
    # R = extrinsics[0:3,0:3]
    # R2 = np.linalg.inv(R)
    # extrinsics[0:3, 0:3] = R2  # this cord except for Rwc
    # intrinsics: line [7), 1x3 matrix
    intrinsics = np.fromstring(' '.join(lines[6:7]), dtype=np.float32, sep=' ')
    depthrange = np.fromstring(' '.join(lines[8:9]), dtype=np.float32, sep=' ')            # depth range
    return intrinsics, extrinsics, depthrange


def View_Selection(view_file, camera_folder, out_file, view_num):
    with open(view_file) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]

    # view selection
    score = np.zeros((len(scans), len(scans)))
    queue = []
    for i in range(len(scans)):
        for j in range(i + 1, len(scans)):
            queue.append((i, j))

    def calc_score(inputs):
        i, j = inputs
        scan_id1 = scans[i]
        scan_id2 = scans[j]
        camera_file1 = os.path.join(camera_folder, '{}.txt'.format(scan_id1))
        camera_file2 = os.path.join(camera_folder, '{}.txt'.format(scan_id2))
        intrinsics1, extrinsics1, depthrange1 = read_camera_parameters(camera_file1)
        intrinsics2, extrinsics2, depthrange2 = read_camera_parameters(camera_file2)
        cam_center_i = extrinsics1[0:3, 3]
        cam_center_j = extrinsics2[0:3, 3]
        cam_rotate_i = extrinsics1[0:3, 0:3]  # expect for Rwc
        cam_rotate_j = extrinsics2[0:3, 0:3]
        depthmid1 = (depthrange1[1] + depthrange1[0]) / 2
        depthmid2 = (depthrange2[1] + depthrange2[0]) / 2
        if args.Cams_ori == 1:
            depthmid1 = -depthmid1 # XrightYup
            depthmid2 = -depthmid2
        world_center_i = np.matmul(cam_rotate_i , [0, 0, depthmid1]) + cam_center_i
        world_center_j = np.matmul(cam_rotate_j , [0, 0, depthmid2]) + cam_center_j
        yweight = 1
        score = np.sqrt((world_center_i[0] - world_center_j[0]) ** 2 + ((world_center_i[1] - world_center_j[1]) * yweight) ** 2)
        return i, j, score

    results = []
    for input in queue:
        result = calc_score(input)
        results.append(result)

    for i, j, s in results:
        score[i, j] = s
        score[j, i] = s

    # skip match
    flag = []
    view_sel = []
    for i in range(len(scans)):
        sorted_score = np.argsort(score[i])
        ref = sorted_score[0]
        if ref not in flag:
            view_sel.append([(scans[k], score[i, k]) for k in sorted_score[0:view_num]])
            for k in sorted_score[1:3]:
                flag.append(k)

    """
    # all match 
    view_sel = []
    for i in range(len(scans)):
        sorted_score = np.argsort(score[i])
        view_sel.append([(scans[k], score[i, k]) for k in sorted_score[0:view_num]])
    """

    with open(out_file, 'w') as f:
        f.write('%d\n' % len(view_sel))
        f.write('%d\n' % view_num)
        for i, sorted_score in enumerate(view_sel):
            for image_id, s in sorted_score:
                f.write('%s %.3f ' % (image_id, s))
            f.write('\n')

    print('Matching pairs saved.')


if __name__ == '__main__':


    image_folder = os.path.join(args.dense_folder, 'Images')
    camera_folder = os.path.join(args.dense_folder, 'Cams')
    view_file = args.dense_folder + '/view.txt'
    out_file = args.dense_folder + '/viewpair.txt'

    # list all view and stored in view.txt
    filenames = sorted(os.listdir(image_folder))
    with open(view_file,'w') as f:
        for i in range(np.size(filenames)):
            fn = filenames[i]
            name = fn[0:len(fn) - 4]
            f.write(str(name))
            f.write('\n')

    # select matching pairs
    View_Selection(view_file, camera_folder, out_file, args.view_num)

