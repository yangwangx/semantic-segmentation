#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
PYTHONPATH=$PWD:$PYTHONPATH python3 eval.py \
	--dataset cityscapes \
    --arch network.deepv3.DeepWV3Plus \
    --inference_mode sliding \
    --scales 1.0 \
    --split val \
    --cv_split 2 \
    --dump_images \
    --crop_size 512 \
    --ckpt_path ${2} \
    --snapshot ${1}
