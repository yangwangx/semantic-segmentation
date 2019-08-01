import os, sys
import numpy as np
from skimage.util import img_as_float
from skimage.color import rgb2lab

import torch
import torch.nn as nn
import torch.nn.functional as FF
from network.mynn import Upsample

svx_root = '/private/home/yangwangx/TemPytorch/standalone/svx_supervox'
sys.path.insert(0, svx_root)
import lib as svx

def get_ssn_pix_model():
    ssn = svx.SSN(use_cnn=False)
    return ssn

def get_ssn_cityscape_model():
    pretrained_npz = os.path.join(svx_root, 'ssn_pretrained/ssn_cityscapes_model.caffemodel.npz')
    ssn = svx.SSN(use_cnn=True, num_in=5, num_out=15, num_ch=64, pretrained_npz=pretrained_npz)
    return ssn

def convert_01_tensor_rgb2lab(imgs):
    # assume imgs: shape (B, 3, H, W), value [0, 1], dtype float32
    return  torch.from_numpy(
                np.asarray(list(
                    np.moveaxis(rgb2lab(np.moveaxis(img.cpu().numpy(), 0, 2)), 2, 0) 
                for img in imgs))
            ).type_as(imgs).to(imgs.device)

def run_ssn_cityscape_model_on_lab(ssn, imgs_lab, num_spixel):
    with torch.no_grad():
        _config = ssn.module.configure if hasattr(ssn, 'module') else ssn.configure
        H, W, Kh, Kw, K = _config(imgs_lab.shape, num_spixel, p_scale=0.4, lab_scale=0.26, softscale=-1.0, num_steps=10)
        init_spIndx = svx.get_init_spIndx2d(imgs_lab.shape, n_sv=num_spixel).to(imgs_lab.device)
        _, _, final_assoc, final_spIndx = ssn(imgs_lab, init_spIndx)
    return init_spIndx, final_assoc, final_spIndx, (Kh, Kw)

def run_ssn_cityscape_model_on_rgb(ssn, imgs_rgb, num_spixel):
    with torch.no_grad():
        # convert rgb to lab format
        imgs_lab = convert_01_tensor_rgb2lab(imgs_rgb)
        return run_ssn_cityscape_model_on_lab(ssn, imgs_lab, num_spixel)

def hard_spix_gather_smear(pFeat, final_spIndx, spShape):
    with torch.no_grad():
        _, _, H1, W1 = final_spIndx.shape
        _, _, H2, W2 = pFeat.shape
        K = final_spIndx.max().item() + 1
        if H1 == H2 and W1 == W2:
            spFeat, _ = svx.spFeatGather2d(pFeat, final_spIndx, K)
            pFeat = svx.spFeatSmear2d(spFeat, final_spIndx)
        else:
            pFeat = Upsample(pFeat, size=(H1, W1))  # upsample by interp
            spFeat, _ = svx.spFeatGather2d(pFeat, final_spIndx, K)
            pFeat = svx.spFeatSmear2d(spFeat, final_spIndx)
            pFeat = Upsample(pFeat, size=(H2, W2))  # downsample by interp
        return pFeat

def downsoft_spix_gather_smear(pFeat, init_spIndx, psp_assoc, spShape):
    with torch.no_grad():
        _, _, H1, W1 = init_spIndx.shape
        _, _, H2, W2 = pFeat.shape
        Kh, Kw = spShape
        K = Kh * Kw
        if H1 == H2 and W1 == W2:
            spFeat, _ = svx.spFeatUpdate2d(pFeat, psp_assoc, init_spIndx, Kh, Kw)
            pFeat = svx.spFeatSoftSmear2d(spFeat, psp_assoc, init_spIndx, Kh, Kw)
        else:
            init_spIndx = Upsample(init_spIndx, size=(H2, W2), mode='nearest')  # downsample by interp
            psp_assoc = Upsample(psp_assoc, size=(H2, W2), mode='nearest')  # downsample by interp
            spFeat, _ = svx.spFeatUpdate2d(pFeat, psp_assoc, init_spIndx, Kh, Kw)
            pFeat = svx.spFeatSoftSmear2d(spFeat, psp_assoc, init_spIndx, Kh, Kw)
        return pFeat

def soft_spix_gather_smear(pFeat, init_spIndx, psp_assoc, spShape):
    with torch.no_grad():
        _, _, H1, W1 = init_spIndx.shape
        _, _, H2, W2 = pFeat.shape
        Kh, Kw = spShape
        K = Kh * Kw
        if H1 == H2 and W1 == W2:
            spFeat, _ = svx.spFeatUpdate2d(pFeat, psp_assoc, init_spIndx, Kh, Kw)
            pFeat = svx.spFeatSoftSmear2d(spFeat, psp_assoc, init_spIndx, Kh, Kw)
        else:
            pFeat = Upsample(pFeat, size=(H1, W1))  # upsample by interp
            spFeat, _ = svx.spFeatUpdate2d(pFeat, psp_assoc, init_spIndx, Kh, Kw)
            pFeat = svx.spFeatSoftSmear2d(spFeat, psp_assoc, init_spIndx, Kh, Kw)
            pFeat = Upsample(pFeat, size=(H2, W2))  # downsample by interp
        return pFeat

def none_spix_gather_smear(pFeat, init_spIndx, spShape):
    with torch.no_grad():
        _, _, H1, W1 = init_spIndx.shape
        _, _, H2, W2 = pFeat.shape
        if H1 == H2 and W1 == W2:
            pass
        else:
            pFeat = Upsample(pFeat, size=(H1, W1))  # upsample by interp
            pFeat = Upsample(pFeat, size=(H2, W2))  # downsample by interp
        return pFeat

def softmax_spix_gather_smear(pFeat, init_spIndx, psp_assoc, spShape):
    with torch.no_grad():
        _, _, H1, W1 = init_spIndx.shape
        _, _, H2, W2 = pFeat.shape
        Kh, Kw = spShape
        K = Kh * Kw
        if H1 == H2 and W1 == W2:
            spFeat, _ = svx.spFeatUpdate2d(pFeat, psp_assoc, init_spIndx, Kh, Kw)
            pFeat = torch.max(pFeat, svx.spFeatSoftSmear2d(spFeat, psp_assoc, init_spIndx, Kh, Kw))
        else:
            pFeat = Upsample(pFeat, size=(H1, W1))  # upsample by interp
            spFeat, _ = svx.spFeatUpdate2d(pFeat, psp_assoc, init_spIndx, Kh, Kw)
            pFeat = torch.max(pFeat, svx.spFeatSoftSmear2d(spFeat, psp_assoc, init_spIndx, Kh, Kw))
            pFeat = Upsample(pFeat, size=(H2, W2))  # downsample by interp
        return pFeat

def hardmax_spix_gather_smear(pFeat, final_spIndx, spShape):
    with torch.no_grad():
        _, _, H1, W1 = final_spIndx.shape
        _, _, H2, W2 = pFeat.shape
        K = final_spIndx.max().item() + 1
        if H1 == H2 and W1 == W2:
            spFeat, _ = svx.spFeatGather2d(pFeat, final_spIndx, K)
            pFeat = torch.max(pFeat, svx.spFeatSmear2d(spFeat, final_spIndx))
        else:
            pFeat = Upsample(pFeat, size=(H1, W1))  # upsample by interp
            spFeat, _ = svx.spFeatGather2d(pFeat, final_spIndx, K)
            pFeat = torch.max(pFeat, svx.spFeatSmear2d(spFeat, final_spIndx))
            pFeat = Upsample(pFeat, size=(H2, W2))  # downsample by interp
        return pFeat

def spix_pool(pFeat, init_spIndx, psp_assoc, final_spIndx, mode, spShape):
    if mode == 'hard':
        return hard_spix_gather_smear(pFeat, final_spIndx, spShape)
    elif mode == 'soft':
        return soft_spix_gather_smear(pFeat, init_spIndx, psp_assoc, spShape)
    elif mode == 'downsoft':
        return downsoft_spix_gather_smear(pFeat, init_spIndx, psp_assoc, spShape)
    elif mode == 'none':
        return none_spix_gather_smear(pFeat, init_spIndx, spShape)
    elif mode == 'softmax':
        return softmax_spix_gather_smear(pFeat, init_spIndx, psp_assoc, spShape)
    elif mode == 'hardmax':
        return hardmax_spix_gather_smear(pFeat, final_spIndx, spShape)
    else:
        raise NotImplementedError
