#!/usr/bin/env bash
echo "Running inference on pretrained_models/cityscapes_best.pt"
PYTHONPATH=$PWD:$PYTHONPATH python3 yw_eval_with_ssn.py \
    --dataset cityscapes \
    --arch network.deepv3.DeepWV3Plus \
    --inference_mode sliding \
    --scales 1.0 \
    --split val \
    --cv_split 2 \
    --snapshot pretrained_models/cityscapes_best.pth \
    --no_flip \
    --crop_size 512 \
    --smear_layer out \
    --smear_mode soft \
    --crop_spixel 1000 \
    --ckpt_path yw_results/split2/out_soft_1000 \
    --dump_images
