import argparse
import collections
import datetime
import logging
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import utils.loss_func as L
from dataset.VMD import VideoMattingDataset
from models.model import FullModel_VMD
from utils.utils import print_loss_dict

BASE_SEED = 777

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(FullModel_VMD.ARCH_DICT.keys()), help='Models')
    parser.add_argument('--data', required=True, help='Path to LMDB dataset')
    parser.add_argument('--load', help='resume from a checkpoint with optimizer parameter attached')
    parser.add_argument('--n_threads', type=int, default=16, help='number of workers for dataflow.')
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--save', default=None)
    parser.add_argument('--trimap', required=True, choices=['narrow', 'medium', 'wide'])
    parser.add_argument('--agg_window', default=7)
    args = parser.parse_args()
    return args

def forward_pretrain(MODEL, dp):
    fg, bg, a, idx = dp      # [B, 3, 3 or 1, H, W]
    loss = {}
    with torch.no_grad():
        out = MODEL(a, fg, bg)
    loss['L_alpha'] = out[0].sum().item()
    loss['L_comp'] = out[1].sum().item()
    loss['L_grad'] = out[2].sum().item()
    loss['L_dt'] = out[3].sum().item()
    loss['L_att'] = out[4].sum().item()
    total = sum(loss.values())
    loss['L_total'] = total

    return [*out[5:], loss, idx]

def main(args):
    logging.basicConfig(level=logging.INFO)
    ########################## set output log dir, write config yml, set summary writer

    if args.save is None:
        args.save = 'results/{}/{}/{}'.format(
            'vmd_subset' if args.subset else 'vmd',
            args.trimap,
            os.path.splitext(args.load)[0]
        )
    os.makedirs(args.save, exist_ok=True)
    outdir = args.save

    ########################## init net, deal with sync_bn
    if args.trimap == 'narrow':
        dilate_kernel = 5   # width: 11
    elif args.trimap == 'medium':
        dilate_kernel = 12  # width: 25
    elif args.trimap == 'wide':
        dilate_kernel = 20  # width: 41
    model = FullModel_VMD(model='vmn_' + args.model, dilate_kernel=dilate_kernel, \
                          agg_window=args.agg_window)

    ########################## load weight if specified
    dct = torch.load(args.load, map_location=torch.device('cpu'))
    missing_keys, unexpected_keys = model.NET.load_state_dict(dct, strict=False)
    print ('Missing keys: ' + str(sorted(missing_keys)))
    print ('Unexpected keys: ' + str(sorted(unexpected_keys)))
    print('Model loaded from', args.load)
    net = nn.DataParallel(model.cuda())

    ########################## setting up dataflow
    eval_dataset = VideoMattingDataset(
        data_root=args.data,
        image_shape=(1088, 1920),   # 1080 % 32 != 0
        mode='val',
        use_subset=args.subset,
        plus1=args.model.startswith('vmn_res'),
        precomputed_val=args.data,
        sample_length=3,
        no_flow=True
    )
    eval_loader = DataLoader(eval_dataset,
                             batch_size=torch.cuda.device_count(),
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=args.n_threads)   

    ########################## main train loop
    vis_loss = collections.OrderedDict()
    h, w = (1080, 1920)
    c = eval_dataset.sample_length // 2

    ### evaluation
    print ('Start evaluation process...')
    net.eval() # mode switch
    eval_loss = {'L_alpha':0., 'L_comp':0., 'L_grad':0.,
                 'L_dt':0., 'L_att':0., 'L_total':0.}
    with torch.no_grad():
        with tqdm(eval_loader, ascii=True) as t:
            for _step, dp in enumerate(t):
                b = dp[0].shape[0]
                out = forward_pretrain(net, dp)
                _, tris, alphas, _, _, _, _, loss, idx = out
                for key in sorted(loss.keys()):
                    vis_loss[key] = '{:.4f}'.format(loss[key] / b)
                    eval_loss[key] += loss[key]

                alphas = alphas[:, c, :, :h, :w]
                tris = tris[:, c, :, :h, :w]
                
                for i in range(b):
                    _id = idx[i]
                    fn = os.path.splitext(eval_dataset.samples[_id][c])[0]
                    os.makedirs(os.path.join(args.save, os.path.dirname(fn)), exist_ok=True)
                    alpha = np.uint8(alphas[i].squeeze().detach().cpu().numpy() * 255)
                    tri = np.uint8(tris[i].squeeze().detach().cpu().numpy() * 255)
                    cv2.imwrite(os.path.join(args.save, fn+'_tri.png'), tri)
                    cv2.imwrite(os.path.join(args.save, fn+'_pred.png'), alpha)

                t.set_postfix(vis_loss)
    for key in eval_loss.keys():
        eval_loss[key] /= float(len(eval_dataset))
    print_loss_dict(eval_loss, os.path.join(outdir, 'loss.log'))

if __name__ == '__main__':
    args = parser()
    main(args)
