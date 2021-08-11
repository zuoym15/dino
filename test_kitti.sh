python eval_kitti.py --data_path /projects/katefgroup/datasets/kitti_tracking/training \
--output_dir ./results_kitti_ft --patch_size 16 \
--pretrained_weights ./checkpoints/dino_deitsmall16_kitti_ft.pth \
--size_mask_neighborhood 6
# --pretrained_weights ./checkpoints/dino_deitsmall16_pretrain_full_checkpoint.pth
# --pretrained_weights ./checkpoints/dino_deitsmall8_pretrain.pth