#!/usr/bin/env bash
echo "Running inference on pretrained_models/cityscapes_best.pt"
PYTHONPATH=$PWD:$PYTHONPATH python3 eval.py \
	--dataset cityscapes \
    --arch network.deepv3.DeepWV3Plus \
    --inference_mode sliding \
    --scales 1.0 \
    --split val \
    --cv_split 2 \
    --crop_size 512 \
    --snapshot pretrained_models/cityscapes_best.pth \
    --no_flip \
    --ckpt_path results/split2_noflip \
    # --dump_images
