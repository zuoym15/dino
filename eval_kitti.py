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

import utils
import vision_transformer as vits

def get_iou(pred, gt, obj_id):
    gt_pos = (gt==obj_id).astype(np.float32)
    pred_pos = (pred==obj_id).astype(np.float32)

    if np.sum(gt_pos) > 0:
        intersection = np.sum(gt_pos*pred_pos)
        union = np.sum(gt_pos) + np.sum(pred_pos) - intersection
        return intersection / union
    else:
        return -1.0 # invalid


def find_subdirs(d):
    return [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

@torch.no_grad()
def eval_video_tracking_davis(args, model, frame_list, video_dir, first_seg, seg_ori, color_palette, seg_table):
    """
    Evaluate tracking on a video given first frame & segmentation
    """
    video_folder = os.path.join(args.output_dir, video_dir.split('/')[-1])
    os.makedirs(video_folder, exist_ok=True)

    # The queue stores the n preceeding frames
    que = queue.Queue(args.n_last_frames)

    # first frame
    frame1, ori_h, ori_w = read_frame(frame_list[0], scale_size=[360])
    # extract first frame feature
    frame1_feat = extract_feature(model, frame1).T #  dim x h*w

    mious = [1.0]
    # saving first segmentation
    out_path = os.path.join(video_folder, "000000.png")
    imwrite_indexed(out_path, seg_ori, color_palette)
    mask_neighborhood = None
    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar = read_frame(frame_list[cnt], scale_size=[360])[0]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]
        
        # exp: try not to use frame0
        # if len(used_frame_feats) > 1:
        #     used_frame_feats = used_frame_feats[1:]
        #     used_segs = used_segs[1:]


        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(args, model, frame_tar, used_frame_feats, used_segs, mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == args.n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=args.patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg_np = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg_np = np.array(frame_tar_seg_np.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(Image.fromarray(frame_tar_seg_np).resize((ori_w, ori_h), 0))
        frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)

        # load gt seg
        seg_path = frame_list[cnt].replace("image_02", "instances")
        gt_seg, _, _ = read_seg_kitti(seg_path, args.patch_size, seg_table=seg_table, scale_size=[360], one_hot=False)
        ious = []
        for i, obj_id in seg_table.items(): # iter through all objs
            pred_seg = np.array((torch.max(norm_mask(seg[0]), dim=0)[1]).squeeze().cpu(), dtype=np.uint8)
            iou = get_iou(pred_seg, gt_seg, i)
            if iou >= 0.0:
                ious.append(iou)

        miou = np.mean(ious)
        mious.append(miou)
        # print('miou', miou)
    return mious

def restrict_neighborhood(h, w):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * args.size_mask_neighborhood + 1):
                for q in range(2 * args.size_mask_neighborhood + 1):
                    if i - args.size_mask_neighborhood + p < 0 or i - args.size_mask_neighborhood + p >= h:
                        continue
                    if j - args.size_mask_neighborhood + q < 0 or j - args.size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - args.size_mask_neighborhood + p, j - args.size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask


def label_propagation(args, model, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    ## we only need to extract feature of the target frame
    feat_tar, h, w = extract_feature(model, frame_tar, return_h_w=True)

    return_feat_tar = feat_tar.T # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)

    if args.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
            mask_neighborhood[:, 0] = 1.0 # the feature at frame0 can be anywhere
            
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    return seg_tar, return_feat_tar, mask_neighborhood
 

def extract_feature(model, frame, return_h_w=False):
    """Extract one frame feature everytime."""
    out = model.get_intermediate_layers(frame.unsqueeze(0).cuda(), n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame.shape[1] / model.patch_embed.patch_size), int(frame.shape[2] / model.patch_embed.patch_size)
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    out = out.reshape(-1, dim)
    if return_h_w:
        return out, h, w
    return out


def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
      raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))+glob.glob(os.path.join(video_dir,"*.png"))]
    frame_list = sorted(frame_list)
    return frame_list


def read_frame(frame_dir, scale_size=[360]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w


def read_seg_kitti(seg_dir, factor, seg_table={}, scale_size=[360], one_hot=True):
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

    seg = Image.fromarray(new_seg_)

    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if(_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 64) * 64)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 64) * 64)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
    if one_hot:
        small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
        small_seg = to_one_hot(small_seg)
    return small_seg, np.asarray(seg), seg_table

def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


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

    # building network
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    color_palette = []
    for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
        color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)

    # video_list = open(os.path.join(args.data_path, "ImageSets/2017/val.txt")).readlines()
    video_list = sorted(find_subdirs(os.path.join(args.data_path, "image_02"))) # 0000 - 0020 []

    max_num_frames = 20
    mious_over_time = np.zeros(max_num_frames)
    video_count = 0

    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
        video_dir = os.path.join(args.data_path, "image_02", video_name)
        frame_list = read_frame_list(video_dir)
        frame_list = frame_list[0:max_num_frames] # now we only seg first 20 frames

        seg_path = frame_list[0].replace("image_02", "instances").replace("jpg", "png")
        first_seg, seg_ori, seg_table = read_seg_kitti(seg_path, args.patch_size, seg_table={}, scale_size=[360])

        if first_seg.shape[1] > 1: # at least one fg class
            video_count += 1

            mious = eval_video_tracking_davis(args, model, frame_list, video_dir, first_seg, seg_ori, color_palette, seg_table)
            mious_over_time += mious
            
            # save a video
            writer = imageio.get_writer(os.path.join(args.output_dir, video_dir.split('/')[-1], 'demo.mp4'), fps=5)
            for i, frame_path in enumerate(frame_list):
                raw_im = imageio.imread(frame_path) # e.g. /projects/katefgroup/datasets/kitti_tracking/training/image_02/0001/000000.png
                raw_im = cv2.resize(raw_im, (1248, 352))
                seg = imageio.imread(os.path.join(args.output_dir, video_dir.split('/')[-1], os.path.basename(frame_path)))
                seg = cv2.resize(seg, (1248, 352))
                if len(seg.shape) == 2:
                    seg = np.repeat(seg[..., None], 3, axis=-1)

                cat_im = np.concatenate((raw_im, seg[..., 0:3]), axis=0)
                cat_im = cv2.putText(cat_im, 'mIoU: %.1f' % (100.*mious[i]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                writer.append_data(cat_im)

            writer.close()
        
        else:
            print('skipping %s: no obj' % video_name)

    mious_over_time /= video_count # mean across vides
    print('miou across the dataset: ', mious_over_time)
