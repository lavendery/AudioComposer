import os
import sys
import math
import numpy as np
import torch
import logging
import pandas as pd
import glob
logger = logging.getLogger(f'main.{__name__}')
sys.path.insert(0, '.')


def collate_1d_or_2d(values, pad_idx=0, left_pad=False, shift_right=False,min_len = None, max_len=None,min_factor=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right,min_len, max_len,min_factor, shift_id)
    else:
        return collate_2d(values, pad_idx, left_pad, shift_right, min_len, max_len, min_factor)

def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False,min_len=None, max_len=None,min_factor=None, shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    if max_len: 
        size = min(size,max_len)
    if min_len:
        size = max(size,min_len)
    if min_factor and (size % min_factor!=0):# size must be the multiple of min_factor
        size += (min_factor - size % min_factor)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f"dst shape:{dst.shape} src shape:{src.shape}"
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, min_len=None,max_len=None,min_factor=None):
    """Collate 2d for melspec,Convert a list of 2d tensors into a padded 3d tensor,pad in mel_length dimension. 
        values[0] shape: (melbins,mel_length)
    """
    size = max(v.shape[1] for v in values) # if max_len is None else max_len
    if max_len: 
        size = min(size,max_len)
    if min_len:
        size = max(size,min_len)
    if min_factor and (size % min_factor!=0):# size must be the multiple of min_factor
        size += (min_factor - size % min_factor)

    if isinstance(values,np.ndarray):
        values = torch.FloatTensor(values)
    if isinstance(values,list):
        values = [torch.FloatTensor(v) for v in values]
    res = torch.ones(len(values), values[0].shape[0],size).to(dtype=torch.float32)*pad_idx
    
    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f"dst shape:{dst.shape} src shape:{src.shape}"
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v[:,:size], res[i][:,size - v.shape[1]:] if left_pad else res[i][:,:v.shape[1]])
    return res


def collate_1d_or_2d_tile(values, shift_right=False,min_len = None, max_len=None,min_factor=None, shift_id=1):
    if len(values[0].shape) == 1:
        return collate_1d_tile(values, shift_right,min_len, max_len,min_factor, shift_id)
    else:
        return collate_2d_tile(values, shift_right,min_len,max_len,min_factor)

def collate_1d_tile(values, shift_right=False,min_len=None, max_len=None,min_factor=None,shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    if max_len: 
        size = min(size,max_len)
    if min_len:
        size = max(size,min_len)
    if min_factor and (size%min_factor!=0):# size must be the multiple of min_factor
        size += (min_factor - size % min_factor)
    res = values[0].new(len(values), size)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f"dst shape:{dst.shape} src shape:{src.shape}"
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        n_repeat = math.ceil((size + 1) / v.shape[0])
        v = torch.tile(v,dims=(1,n_repeat))[:size]
        copy_tensor(v, res[i])

    return res


def collate_2d_tile(values, shift_right=False, min_len=None,max_len=None,min_factor=None):
    """Collate 2d for melspec,Convert a list of 2d tensors into a padded 3d tensor,pad in mel_length dimension. """
    size = max(v.shape[1] for v in values) # if max_len is None else max_len
    if max_len: 
        size = min(size,max_len)
    if min_len:
        size = max(size,min_len)
    if min_factor and (size % min_factor!=0):# size must be the multiple of min_factor
        size += (min_factor - size % min_factor)

    if isinstance(values,np.ndarray):
        values = torch.FloatTensor(values)
    if isinstance(values,list):
        values = [torch.FloatTensor(v) for v in values]
    res = torch.zeros(len(values), values[0].shape[0],size).to(dtype=torch.float32)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        n_repeat = math.ceil((size + 1) / v.shape[1])
        v = torch.tile(v,dims=(1,n_repeat))[:,:size]
        copy_tensor(v, res[i])
        
    return res