#!/usr/bin/env bash

    # Example on Cityscapes
     python -m torch.distributed.launch --nproc_per_node=4 train.py \
        --dataset cityscapes \
        --cv 2 \
        --arch network.deepv3.DeepWV3Plus \
        --snapshot ./pretrained_models/cityscapes_best.pth \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --maxSkip 3 \
        --max_cu_epoch 120 \
        --lr 0.001 \
        --lr_schedule scl-poly \
        --poly_exp 1.0 \
        --repoly 1.5  \
        --rescale 1.0 \
        --syncbn \
        --sgd \
        --crop_size 800 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --color_aug 0.25 \
        --gblur \
        --max_epoch 180 \
        --coarse_boost_classes 14,6,15,16,3,12,17,4 \
        --jointwtborder \
        --strict_bdr_cls 5,6,7,11,12,17,18 \
        --rlx_off_epoch 120 \
        --wt_bound 1.0 \
        --bs_mult 2 \
        --apex \
        --exp cv2_skip3_wt1_relax_epoch180_run \
        --ckpt ./logs/ \
        --tb_path ./logs/
