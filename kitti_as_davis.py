# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Some parts are taken from https://github.com/Liusifei/UVC
"""
import os
import copy
import glob
import queue
from urllib.request import urlopen
import argparse
import numpy as np
from tqdm import tqdm

import cv2
import imageio
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
import shutil

import utils

def find_subdirs(d):
    return [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))+glob.glob(os.path.join(video_dir,"*.png"))]
    frame_list = sorted(frame_list)
    return frame_list

def read_seg_kitti(seg_dir, seg_table={}):
    seg = Image.open(seg_dir)
    seg = np.asarray(seg) # get instance ids
    obj_ids = np.unique(seg)
    new_seg_ = np.zeros_like(seg, dtype=np.uint8) # 0 reserved for bkg
    
    if len(seg_table) == 0: # create a new one for first frame
        for i, obj_id in enumerate(obj_ids):
            if obj_id != 10000: # ignore
                new_seg_[seg==obj_id] = i
                seg_table[i] = obj_id
    else: # re-use the table from last frame, for id consistency
        for i, obj_id in seg_table.items():
            new_seg_[seg==obj_id] = i

    return new_seg_, seg_table

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with video object segmentation on DAVIS 2017')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--output_dir', default=".", help='Path where to save segmentations')
    parser.add_argument('--data_path', default='/path/to/davis/', type=str)
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    parser.add_argument("--bs", type=int, default=6, help="Batch size, try to reduce if OOM")
    args = parser.parse_args()

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    color_palette = []
    for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
        color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)

    # video_list = open(os.path.join(args.data_path, "ImageSets/2017/val.txt")).readlines()
    video_list = sorted(find_subdirs(os.path.join(args.data_path, "image_02"))) # 0000 - 0020 []

    max_num_frames = 20
    mious_over_time = np.zeros(max_num_frames)
    video_count = 0

    davis_raw_imgs_dir = './kitti_as_davis/JPEGImages/480p'
    davis_segs_dir = './kitti_as_davis/Annotations/480p'
    davis_meta_dir = './kitti_as_davis/ImageSets/2017/val.txt'

    with open(davis_meta_dir, 'w') as f: # clean up
        pass

    for i, video_name in enumerate(video_list):
        if not os.path.exists(os.path.join(davis_raw_imgs_dir, video_name)):
            os.makedirs(os.path.join(davis_raw_imgs_dir, video_name))
        if not os.path.exists(os.path.join(davis_segs_dir, video_name)):
            os.makedirs(os.path.join(davis_segs_dir, video_name))

        video_name = video_name.strip()
        print(f'[{i}/{len(video_list)}] Begin to parse video {video_name}.')
        video_dir = os.path.join(args.data_path, "image_02", video_name)
        frame_list = read_frame_list(video_dir)
        frame_list = frame_list[0:max_num_frames] # now we only seg first 20 frames

        seg_table = {} # empty for frame0 

        ok = True 
        for i, frame_path in enumerate(frame_list):
            seg_path = frame_list[i].replace("image_02", "instances").replace("jpg", "png")
            seg, seg_table = read_seg_kitti(seg_path, seg_table)

            if len(np.unique(seg))==1: # no fg obj
                if i==0:
                    ok = False
                    print('skipping %s: no obj' % video_name)
                else:
                    print('early stop at frame %d for video %s: no obj' % (i, video_name))
                break

            im = Image.fromarray(seg)
            im.putpalette(color_palette.ravel())
            im.save(os.path.join(davis_segs_dir, video_name, os.path.basename(frame_path)), format='PNG')

            shutil.copyfile(frame_path, os.path.join(davis_raw_imgs_dir, video_name, os.path.basename(frame_path))) # copy raw file

        if ok:
            with open(davis_meta_dir, 'a') as f:
                f.write('%s\n'%video_name) 
