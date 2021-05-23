import argparse
import collections
import datetime
import logging
import multiprocessing as mp
import os
import sys

import cv2 as cv
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
from dataset.DIM import DIMEvalDataset
from dataset.VMD import VideoMattingDataset
from models.model import FullModel, FullModel_VMD
from utils.utils import print_loss_dict

BASE_SEED = 777

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Models')
    parser.add_argument('--dataset', required=True, choices=['dim', 'vmd', 'vmd_subset'])
    parser.add_argument('--data', required=True, help='Path to LMDB dataset')
    parser.add_argument('--load', help='resume from a checkpoint with optimizer parameter attached')
    parser.add_argument('--n_threads', type=int, default=16, help='number of workers for dataflow.')
    parser.add_argument('--trimap', required=True, choices=['narrow', 'medium', 'wide'])
    parser.add_argument('--save', default=None)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--agg_window', default=7)
    args = parser.parse_args()
    return args

def SAD(a, g, m):
    ca = np.float32(a) / 255.0
    cg = np.float32(g) / 255.0
    return np.mean(np.abs(ca[m] - cg[m]))

def MSE(a, g, m):
    ca = np.float32(a) / 255.0
    cg = np.float32(g) / 255.0
    return np.mean((ca[m] - cg[m]) ** 2)

def forward_pretrain(MODEL, a, fg, bg, sub_losses):
    loss = {}
    with torch.no_grad():
        out = MODEL(a, fg, bg)
    loss['L_total'] = 0.
    for i in range(3):
        v = out[i].sum().item()
        loss[sub_losses[i]] = v
        loss['L_total'] += v

    return [*out[3:], loss]

def main(args):
    logging.basicConfig(level=logging.INFO)

    if args.save is None:
        args.save = 'results/{}_single/{}/{}'.format(
            args.dataset, args.trimap,
            os.path.splitext(args.load)[0]
        )
    os.makedirs(args.save, exist_ok=True)
    outdir = args.save
    if args.vis:
        vis_outdir = os.path.join(args.save, 'vis')
        os.makedirs(vis_outdir, exist_ok=True)

    if args.trimap == 'narrow':
        dilate_kernel = 5   # width: 11
    elif args.trimap == 'medium':
        dilate_kernel = 12  # width: 25
    elif args.trimap == 'wide':
        dilate_kernel = 20  # width: 41
    model = FullModel(model=args.model, dilate_kernel=dilate_kernel,\
                      agg_window=args.agg_window)
    dct = torch.load(args.load, map_location=torch.device('cpu'))
    if 'state_dict' in dct.keys():
        dct = dct['state_dict']
    missing_keys, unexpected_keys = model.NET.load_state_dict(dct, strict=False)
    print ('Missing keys: ' + str(sorted(missing_keys)))
    print ('Unexpected keys: ' + str(sorted(unexpected_keys)))
    print('Model loaded from', args.load)
    net = nn.DataParallel(model.cuda())

    ########################## setting up dataflow
    if args.dataset == 'dim':
        eval_dataset = DIMEvalDataset(data_root=args.data, \
            plus1=args.model.startswith('vmn_res'),
            min_shape=2112,
            val_mode='origin'
        )
    else:
        eval_dataset = VideoMattingDataset(
            data_root=args.data,
            image_shape=(1088, 1920),
            mode='val',
            use_subset=args.dataset.endswith('subset'),
            plus1=args.model.startswith('vmn_res'),
            no_flow=True,
            sample_length=3,
            precomputed_val=args.data,
        )
    eval_loader = DataLoader(eval_dataset,
                             batch_size=torch.cuda.device_count(),
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,
                             num_workers=args.n_threads)
    
    step_per_epoch = len(eval_loader)

    vis_loss = collections.OrderedDict()
    c = 1 if args.dataset.startswith('dim') else eval_dataset.sample_length // 2
    ### evaluation
    print ('Start evaluation process...')
    net.eval() # mode switch
    sub_losses = ['L_alpha', 'L_comp', 'L_grad'] if not args.model.endswith('fba') else \
                 ['L_alpha_comp', 'L_lap', 'L_grad']
    eval_loss = {sub_losses[0]:0., sub_losses[1]:0., sub_losses[2]:0., 'L_total':0., 'mSAD':0., 'MSE':0.}
    with torch.no_grad():
        with tqdm(eval_loader, ascii=True) as tq:
            for _step, dp in enumerate(tq):
                b = dp[-1].shape[0]
                if args.dataset.startswith('dim'):
                    gt, fg, bg, _size, _idx = dp
                else:
                    fg, bg, gt, _idx = dp

                out = forward_pretrain(net, gt, fg, bg, sub_losses)
                _imgs, tris, alphas, _, _, _, _, loss = out
                for key in sorted(loss.keys()):
                    vis_loss[key] = '{:.4f}'.format(loss[key] / b)
                    eval_loss[key] += loss[key]

                if args.dataset.startswith('dim'):
                    g, a, t = [None] * b, [None] * b, [None] * b
                    for i in range(b):
                        h, w = _size[i].numpy()
                        if args.model.endswith('dim'):
                            imsize = np.asarray([h, w], dtype=np.float)
                            new_imsize = np.ceil(imsize / 32) * 32
                            nh, nw = int(new_imsize[0]), int(new_imsize[1])
                            g[i] = cv.imread(os.path.join(eval_dataset.data_root, eval_dataset.sample_fn[_idx[i].item()][1]), cv.IMREAD_GRAYSCALE)[np.newaxis, ...]
                            a[i] = cv.resize(alphas[i, c, 0, :nh, :nw].detach().cpu().numpy(), (w, h), interpolation=cv.INTER_CUBIC)[np.newaxis, ...]
                            t[i] = cv.resize(tris[i, c, 0, :nh, :nw].detach().cpu().numpy(), (w, h), interpolation=cv.INTER_NEAREST)[np.newaxis, ...]
                            #assert a[i].shape[0] == g[i].shape[0] and a[i].shape[1] == g[i].shape[1], '{} {}'.format(a[i].shape, g[i].shape)
                            a[i] = np.uint8(np.clip(a[i] * 255, 0, 255))
                            t[i] = np.uint8(t[i] * 255)
                        else:
                            g[i] = np.uint8(gt[i:i+1, c, 0, :h, :w].detach().cpu().numpy())
                            a[i] = np.uint8(alphas[i:i+1, c, 0, :h, :w].detach().cpu().numpy() * 255)
                            t[i] = np.uint8(tris[i:i+1, c, 0, :h, :w].detach().cpu().numpy() * 255)
                    g = np.concatenate(g, axis=0)
                    a = np.concatenate(a, axis=0)
                    t = np.concatenate(t, axis=0)
                else:
                    a = np.uint8(alphas[:, c, 0, :1080, ...].detach().cpu().numpy() * 255)
                    t = np.uint8(tris[:, c, 0, :1080, ...].detach().cpu().numpy() * 255)
                    g = np.uint8(gt[:, c, 0, :1080, ...].detach().cpu().numpy())

                assert t.shape == a.shape, '{} {} {}'.format(t.shape, a.shape, g.shape)

                m = (t > 0) * (t < 255)
                for i in range(b):
                    h, w = _size[i].numpy() if args.dataset.startswith('dim') else (1080, 1920)
                    sad = SAD(a[i], g[i], m[i])
                    mse = MSE(a[i], g[i], m[i])
                    eval_loss['mSAD'] += sad
                    eval_loss['MSE'] += mse
                    pcount = np.sum(m[i])

                    _id = _idx[i]
                    if args.dataset.startswith('dim'):
                        fn = '{:05d}'.format(_id)
                    else:
                        fn = os.path.splitext(eval_dataset.samples[_id][c])[0]

                    if args.save:
                        os.makedirs(os.path.join(args.save, os.path.dirname(fn)), exist_ok=True)
                        cv.imwrite(os.path.join(args.save, fn+'_tri.png'), t[i])
                        cv.imwrite(os.path.join(args.save, fn+'_pred.png'), a[i])

                    if args.vis:
                        vis_t = cv.resize(t[i], (w // 4, h // 4), interpolation=cv.INTER_NEAREST)
                        comp = cv.hconcat([a[i], g[i]])
                        _left = comp.shape[1] // 2 - vis_t.shape[1] // 2
                        comp[-vis_t.shape[0]:, _left:_left+vis_t.shape[1]] = vis_t
                        comp = cv.cvtColor(comp, cv.COLOR_GRAY2BGR)
                        comp = cv.copyMakeBorder(comp, 0, 100, 0, 0,
                            cv.BORDER_CONSTANT, value=(255, 0, 0))
                        s = 'SAD={:.6f} MSE={:.6f} valid_pxs={}'.format(\
                                sad, mse, pcount)
                        comp = cv.putText(comp, s, (20, comp.shape[0] - 20), \
                            cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        if not args.dataset.startswith('dim'):
                            os.makedirs(os.path.join(vis_outdir, os.path.dirname(fn)), exist_ok=True)
                        cv.imwrite(os.path.join(vis_outdir, fn+'.png'), comp)

                tq.set_postfix(vis_loss)
    for key in eval_loss.keys():
        eval_loss[key] /= float(len(eval_dataset))
    print_loss_dict(eval_loss, os.path.join(outdir, 'metric.log'))

if __name__ == '__main__':
    args = parser()
    main(args)
