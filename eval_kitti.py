# from numpy import random
# from numpy.core.numeric import full
import torch
import numpy as np
import os
import copy

import utils
import util.geom
import util.improc
import util.py
import cv2
import matplotlib.pyplot as plt
# import scipy.ndimage
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image

import queue

import vision_transformer as vits
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from PIL import Image
# import random
# import utils.py
# import utils.geom
# import utils.improc

# import glob
# import json

color_palette = []
for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
    color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)

def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
      raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x

def prep_frame(img, scale_size=[320]):
    """
    read a single frame & preprocess
    """
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

def restrict_neighborhood(h, w, size_mask_neighborhood=12):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

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

def label_propagation(model, frame_tar, list_frame_feats, list_segs, mask_neighborhood=None, size_mask_neighborhood=12, topk=5):
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

    if size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    # print('segs', segs.shape)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    # print('segs', segs.shape)
    # print('aff', aff.shape)
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    return seg_tar, return_feat_tar, mask_neighborhood




data_dir = '/projects/katefgroup/datasets/kitti/processed/npzs/traj_ah_s2_i1'

print('declaring model; passing to gpu')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
# model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

patch_size=16
if patch_size==8:
    ckpt='./checkpoints/dino_deitsmall8_pretrain.pth'
elif patch_size==16:
    ckpt='./checkpoints/dino_deitsmall16_pretrain_full_checkpoint.pth' 
    # ckpt='./checkpoints/dino_deitsmall16_kitti_ft.pth' 
else:
    assert False, "patch size has to be 8 or 16"

model = vits.__dict__['vit_small'](patch_size=patch_size, num_classes=0)
model.cuda()
utils.load_pretrained_weights(model, ckpt, "teacher", 'vit_small', patch_size)
for param in model.parameters():
    param.requires_grad = False
model.eval()
model.to(device)

n_last_frames = 7
mask_neighborhood = None

for seq in [4, 5, 6, 7, 8, 9, 10, 11]:

    target_tid = None

    # The queue stores the n preceeding frames
    que = queue.Queue(n_last_frames)

    num_frames = 100


    rgbs = []
    feats = []
    frames = []
    masks = []
    bin_masks = []

    for fr in range(num_frames):
        fn = 'seq_%04d_startframe_%06d.npz' % (seq, fr)
        filename = '%s/%s' % (data_dir, fn)
        d = np.load(filename, allow_pickle=True)
        d = dict(d)

        rgb_camXs = d['rgb_camXs'] # S, H, W, 3
        # print('rgb_camXs', rgb_camXs.shape)
        # utils.py.print_stats('rgb', rgb_camXs)

        boxlist_camXs = d['boxlists']
        scorelist_s = d['scorelists']
        tidlist_s = d['tidlists']
        pix_T_cams = d['pix_T_cams']

        # print('boxlist_camXs', boxlist_camXs.shape)
        # print('scorelist_s', scorelist_s.shape)


        rgb = rgb_camXs[0]
        H, W, C = rgb.shape
        boxlist = boxlist_camXs[0]
        scorelist = scorelist_s[0]
        tidlist = tidlist_s[0]
        pix_T_cam = pix_T_cams[0]


        if target_tid is None:
            target_tid = tidlist[0]
            
        # boxlist = boxlist[scorelist > 0]
        boxlist = boxlist[tidlist == target_tid]
        # print('boxlist', boxlist.shape)

        if len(boxlist) > 0:

            boxlist = torch.from_numpy(boxlist)
            pix_T_cam = torch.from_numpy(pix_T_cam)

            lrtlist = util.geom.convert_boxlist_to_lrtlist(boxlist.unsqueeze(0))

            masklist = util.geom.get_masklist_from_lrtlist(pix_T_cam.unsqueeze(0), lrtlist, H, W) # 1, 1, 1, H, W
            # print('masklist', masklist.shape)
        else:
            masklist = torch.zeros((1, 1, 1, H, W), dtype=torch.float32)

        masklist_py = masklist[0,0,0].detach().cpu().numpy() # H, W

        if False:
            fname = 'kitti_mask_%d.png' % fr
            plt.imsave(fname=fname, arr=masklist, format='png')
            print('saved', fname)
            print('rgb', rgb.shape)
            fname = 'kitti_rgb_%d.png' % fr
            plt.imsave(fname=fname, arr=rgb, format='png')
            print('saved', fname)


        rgbs.append(rgb)

        # first frame
        frame1, ori_h, ori_w = prep_frame(rgb)

        mask = masklist[0] # 1, 1, H, W
        bin_masks.append(mask)
        mask = F.interpolate(mask, (frame1.shape[1], frame1.shape[2])) # 1, 1, H, W
        mask = F.interpolate(mask, scale_factor=1/patch_size).round().float() # 1, 1, H, W
        mask = to_one_hot(mask[0], n_dims=None)

        frames.append(frame1)
        masks.append(mask)

        # print('frame1', frame1.shape)
        # print('mask', mask.shape)

        if fr==0:
            frame1_feat, h, w = extract_feature(model, frame1, return_h_w=True) #  dim x h*w
            frame1_feat = frame1_feat.T # dim, h*w
            print('frame1_feat', frame1_feat.shape)


            frame1_featmap = frame1_feat.reshape([-1, h, w])
            print('frame1_featmap', frame1_featmap.shape)

            feats.append(frame1_featmap)


    # print('frame1', frame1.shape)

    # # extract first frame feature

    first_feat = feats[0].reshape(-1, h*w)
    first_seg = masks[0]

    for fr in range(num_frames):

        frame_tar = frames[fr]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [first_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        print('used_segs[0]', used_segs[0].shape)
        # print('used_segs[1]', used_segs[1].shape)
        print('len(used_segs)', len(used_segs))

        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(model, frame_tar, used_frame_feats, used_segs, mask_neighborhood)
        print('frame_tar_avg', frame_tar_avg.shape)
        print('feat_tar', feat_tar.shape)
        print('mask_neighborhood', mask_neighborhood.shape)

        # pop out oldest frame if neccessary
        if que.qsize() == n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))

        fname = './kitti_vis/kitti_%02d_prop_%04d.png' % (seq, fr)
        # imwrite_indexed(fname, frame_tar_seg, color_palette)

        rgb = rgbs[fr]
        seg = np.ones_like(rgb)
        # gt = F.interpolate(bin_masks[fr], (ori_h, ori_w))[0,0].detach().cpu().numpy()
        gt = bin_masks[fr][0,0].detach().cpu().numpy()
        # match = frame_tar_seg==gt
        seg[:,:,0] = frame_tar_seg*255
        seg[:,:,1] = gt*128
        seg[:,:,2] = 0

        cat = np.concatenate([rgb, seg], axis=0)
        plt.imsave(fname=fname, arr=cat, format='png')
        print('saved', fname)

    if False:
        feats = torch.stack(feats, dim=0) # S, C, H, W
        print('feats', feats.shape)
        feats_pca = util.improc.get_feat_pca(feats)
        # print('feat_pca', feat_pca.shape)
        feats_pca = F.interpolate(feats_pca, scale_factor=8)

        for fr in range(feats_pca.shape[0]):
            feat_pca = feats_pca[fr]
            feat_pca_py = feat_pca.permute(1,2,0).detach().cpu().numpy() + 0.5
            fname = 'kitti_pca_%d.png' % fr
            plt.imsave(fname=fname, arr=feat_pca_py, format='png')
            print('saved', fname)


            # for j in range(nh):
            #     fname = os.path.join('.', "attn-head" + str(j) + ".png")
            #     plt.imsave(fname=fname, arr=attentions[j], format='png')
                # print(f"{fname} saved.")



