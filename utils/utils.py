import logging
import time
from pathlib import Path
import cv2 as cv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as torch_dist

def dt(a):
    # a: tensor, [B, S, H, W]
    ac = a.cpu().numpy()
    b, s = a.shape[:2]
    z = []
    for i in range(b):
        y = []
        for j in range(s):
            x = ac[i,j]
            y.append(cv.distanceTransform((x * 255).astype(np.uint8), cv.DIST_L2, 0))
        z.append(np.stack(y))
    return torch.from_numpy(np.stack(z)).float().to(a.device)

def trimap_transform(trimap):
    # trimap: tensor, [B, S, 2, H, W]
    b, s, _, h, w = trimap.shape

    clicks = torch.zeros((b, s, 6, h, w), device=trimap.device)
    for k in range(2):
        tk = trimap[:, :, k]
        if torch.sum(tk != 0) > 0:
            dt_mask = -dt(1. - tk)**2
            L = 320
            clicks[:, :, 3*k] = torch.exp(dt_mask / (2 * ((0.02 * L)**2)))
            clicks[:, :, 3*k+1] = torch.exp(dt_mask / (2 * ((0.08 * L)**2)))
            clicks[:, :, 3*k+2] = torch.exp(dt_mask / (2 * ((0.16 * L)**2)))

    return clicks

def torch_barrier():
    if torch_dist.is_initialized():
        torch_dist.barrier()

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    ALL PROCESSES has the averaged results.
    """
    if torch_dist.is_initialized():
        world_size = torch_dist.get_world_size()
        if world_size < 2:
            return inp
        with torch.no_grad():
            reduced_inp = inp
            torch.distributed.all_reduce(reduced_inp)
            torch.distributed.barrier()
        return reduced_inp / world_size
    return inp

def print_loss_dict(loss, save=None):
    s = ''
    for key in sorted(loss.keys()):
        s += '{}: {:.6f}\n'.format(key, loss[key])
    print (s)
    if save is not None:
        with open(save, 'w') as f:
            f.write(s)

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], axis=0)
    return coords.unsqueeze(0).repeat(batch, 1, 1, 1)

def grid_sampler(img, coords, mode='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates
        img:    [B, C, H, W]
        coords: [B, 2, H, W]
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split(1, dim=1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=1).permute(0, 2, 3, 1)
    img = F.grid_sample(img, grid, mode=mode, align_corners=True)

    return img

def flow_dt(a, ha, gt, hgt, flow, trimask, metric=False, cuda=True):
        '''
        All tensors in [B, C, H, W]
        a: current prediction
        gt: current groundtruth
        ha: adjacent frame prediction
        hgt: adjacent frame groundtruth
        flow: optical flow from current frame to adjacent frame
        trimask: current frame trimask
        '''
        # Warp ha back to a and hgt back to gt
        with torch.no_grad():
            B, _, H, W = a.shape
            mask = torch.isnan(flow)               # B, 1, H, W
            coords = coords_grid(B, H, W)          # B, 2, H, W
            if cuda:
                coords = coords.to(torch.cuda.current_device())
            flow[mask] = 0
            flow_coords = coords + flow
            mask = (~mask[:, :1, :, :]) * trimask.bool()
            valid = mask.sum()
            if valid == 0:
                if metric:
                    return valid.float(), valid.float(), valid.float()
                else:
                    return valid.float()
        
        pgt = grid_sampler(hgt, flow_coords)
        pa = grid_sampler(ha, flow_coords)
        error = torch.abs((a[mask] - gt[mask]) - (pa[mask] - pgt[mask]))    # L1 instead of L2
        if metric:
            error2 = torch.abs((a[mask] - gt[mask]) ** 2 - (pa[mask] - pgt[mask]) ** 2)
            return error.sum(), error2.sum(), valid
        return error.mean()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(output_dir, cfg_name, phase='train'):
    root_output_dir = Path(output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    final_output_dir = root_output_dir / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(final_output_dir)

def poly_lr(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

def const_lr(optimizer, base_lr, max_iters, cur_iters):
    return base_lr

OPT_DICT = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
}

STR_DICT = {
    'poly': poly_lr,
    'const': const_lr,
}
