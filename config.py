"""
# Code adapted from:
# https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py

Source License
# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""
##############################################################################
#Config
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch


from utils.attr_dict import AttrDict


cfg = AttrDict()
cfg.EPOCH = 0
# Use Class Uniform Sampling to give each class proper sampling
cfg.CLASS_UNIFORM_PCT = 0.0

# Use class weighted loss per batch to increase loss for low pixel count classes per batch
cfg.BATCH_WEIGHTING = False

# Border Relaxation Count
cfg.BORDER_WINDOW = 1
# Number of epoch to use before turn off border restriction
cfg.REDUCE_BORDER_EPOCH = -1
# Comma Seperated List of class id to relax
cfg.STRICTBORDERCLASS = None



# Attribute Dictionary for Dataset
cfg.DATASET = AttrDict()
# Cityscapes Dir Location
cfg.DATASET.CITYSCAPES_DIR = '/private/home/yangwangx/datasets/cityscapes'
# SDC Augmented Cityscapes Dir Location
cfg.DATASET.CITYSCAPES_AUG_DIR = ''
# Mapillary Dataset Dir Location
cfg.DATASET.MAPILLARY_DIR = ''
# Kitti Dataset Dir Location
cfg.DATASET.KITTI_DIR = ''
# SDC Augmented Kitti Dataset Dir Location
cfg.DATASET.KITTI_AUG_DIR = ''
# Camvid Dataset Dir Location
cfg.DATASET.CAMVID_DIR = ''
# Number of splits to support
cfg.DATASET.CV_SPLITS = 3

# Attribute Dictionary for Model
cfg.MODEL = AttrDict()
cfg.MODEL.BN = 'regularnorm'
cfg.MODEL.BNFUNC = None

def assert_and_infer_cfg(args, make_immutable=True, train_mode=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """

    if hasattr(args, 'syncbn') and args.syncbn:
        if args.apex:
            import apex
            cfg.MODEL.BN = 'apex-syncnorm'
            cfg.MODEL.BNFUNC = apex.parallel.SyncBatchNorm
        else:
            raise Exception('No Support for SyncBN without Apex')
    else:
        cfg.MODEL.BNFUNC = torch.nn.BatchNorm2d
        print('Using regular batch norm')

    if not train_mode:
        cfg.immutable(True)
        return
    if args.class_uniform_pct:
        cfg.CLASS_UNIFORM_PCT = args.class_uniform_pct

    if args.batch_weighting:
        cfg.BATCH_WEIGHTING = True

    if args.jointwtborder:
        if args.strict_bdr_cls != '':
            cfg.STRICTBORDERCLASS = [int(i) for i in args.strict_bdr_cls.split(",")]
        if args.rlx_off_epoch > -1:
            cfg.REDUCE_BORDER_EPOCH = args.rlx_off_epoch

    if make_immutable:
        cfg.immutable(True)
