import numpy as np
import imageio
import glob
import os


def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list

raw_dir = '/projects/katefgroup/datasets/cater/as_davis/JPEGImages/480p/CLEVR_new_000000/'
seg_dir = '/home/yzuo/dino/results_cater/CLEVR_new_000000/'
writer = imageio.get_writer('cater_vis.mp4', fps=20)
filenames = read_frame_list(raw_dir)

for raw_fn in filenames:
    seg_fn = raw_fn.replace(raw_dir, seg_dir).replace("jpg", "png")
    raw_im = imageio.imread(raw_fn)
    seg_im = imageio.imread(seg_fn)

    im = np.concatenate((raw_im, seg_im[..., 0:3]), axis=1)
    writer.append_data(im)

writer.close()
    