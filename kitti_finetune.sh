python -m torch.distributed.launch --nproc_per_node=1 \
main_dino.py --arch vit_small --patch_size 16 --batch_size_per_gpu 128 \
--data_path /projects/katefgroup/datasets/kitti_tracking/training/ \
--epochs 30 --freeze_last_layer 0 --warmup_epochs 0 --lr 1e-5 --min_lr 1e-5 --saveckp_freq 10 \
--weight_decay 0.4 --weight_decay_end 0.4 \
--resume ./checkpoints/dino_deitsmall16_pretrain_full_checkpoint.pth \
--local_crops_scale 0.4 0.8 --global_crops_scale 1.0 1.5 \
--output_dir ./logs_kitti \
--do_kitti \

