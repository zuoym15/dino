python -m torch.distributed.launch --nproc_per_node=2 \
main_dino.py --arch vit_small --patch_size 16 --batch_size_per_gpu 128 \
--data_path /projects/katefgroup/datasets/cater/raw/aa_s300_c6_m10/images/ --output_dir ./logs_cater \
--epochs 10 --freeze_last_layer 0 --warmup_epochs 0 --lr 1e-6 --min_lr 1e-6 --saveckp_freq 1 \
--weight_decay 0.4 --weight_decay_end 0.4 \
--resume ./checkpoints/dino_deitsmall16_pretrain_full_checkpoint.pth