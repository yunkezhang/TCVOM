import logging
import os
import json
import random
import sys
import time

import cv2
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from imgaug import parameters as iap

from utils.utils import coords_grid, grid_sampler


class VideoMattingDataset(torch.utils.data.Dataset):
    VIDEO_SHAPE = (1080, 1920)
    FLOW_QUANTIZATION_SCALE = 100
    FG_FOLDER = 'FG_done'
    BG_FOLDER = 'BG_done'
    FLOW_FOLDER = 'flow_png'
    def __init__(self, data_root, image_shape, plus1, mode, \
                 use_subset=False, no_flow=False, precomputed_val=None, \
                 sample_length=5):
        self.no_flow = no_flow
        self.mode = mode
        self.precomputed_val = precomputed_val
        self.sample_length = sample_length
        assert self.mode in ['train', 'val']
        if self.precomputed_val is not None:
            assert self.mode == 'val'
        self.data_root = data_root
        if plus1:
            self.image_shape = [image_shape[0]+1, image_shape[1]+1]
        else:
            self.image_shape = list(image_shape)
        setname = '{}_videos_subset.txt' if use_subset else '{}_videos.txt'
        setname = setname.format(self.mode)

        with open(os.path.join(self.data_root, 'frame_corr.json'), 'r') as f:
            self.frame_corr = json.load(f)
        with open(os.path.join(self.data_root, setname), 'r') as f:
            self.samples = self.parse(f)
        #self.samples = self.samples[:240]
        self.dataset_length = len(self.samples)

        # apply to fg, bg
        self.pixel_aug = iaa.Sequential([
            iaa.MultiplyHueAndSaturation(mul=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5)), # mean, std, low, high
            iaa.GammaContrast(gamma=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5)),
            iaa.AddToHue(value=iap.TruncatedNormal(0.0, 0.1*100, -0.2*255, 0.2*255)),
        ])
        self.jpeg_aug = iaa.Sometimes(0.6, iaa.JpegCompression(compression=(70, 99)))

    def __len__(self):
        return self.dataset_length

    def img_crop_and_resize(self, img, ph, pw, nsize=None, mode='bilinear'):
        img2 = img[ph:ph+nsize[0], pw:pw+nsize[1]] if nsize is not None else img
        img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
        img2 = torch.floor(F.interpolate(img2, self.image_shape, mode=mode, align_corners=True) + 0.5)
        return img2
    
    def flow_crop_and_resize(self, flow, ph, pw, nsize=None, mode='bilinear'):
        if nsize is not None:
            flow = flow[ph:ph+nsize[0], pw:pw+nsize[1]]
        else:
            nsize = (flow.shape[0], flow.shape[1])
        flow = flow.permute(2, 0, 1).unsqueeze(0)
        
        # gradient check
        def _gradient_check(_fa, _fb):
            _fa = _fa.squeeze()
            _fb = _fb.squeeze()
            dotab = (_fa * _fb).sum(dim=0)
            nfa = torch.sqrt((_fa ** 2).sum(dim=0))
            nfb = torch.sqrt((_fb ** 2).sum(dim=0))
            nfab = nfa * nfb
            # cosine
            eps = 1e-6
            angle = torch.acos((dotab / nfab).abs().clamp(0, 1.0-eps))
            angle_valid = angle <= np.pi / 4
            angle_valid[nfab == 0] = True
            angle_valid[(nfa + nfb) < 2] = True
            # magnitude
            mag = torch.abs(nfa - nfb)
            mag_valid = mag < 50
            return (angle_valid * mag_valid).int()

        gradh = _gradient_check(flow[:, :, :-1, :], flow[:, :, 1:, :])
        gradh = F.pad(gradh, (0, 0, 0, 1), value=1)
        gradw = _gradient_check(flow[:, :, :, :-1], flow[:, :, :, 1:])
        gradw = F.pad(gradw, (0, 1, 0, 0), value=1)
        valid = gradw * gradh                                           # H, W

        # interpolate
        sample_scaleh = (nsize[0]-1) / float(self.image_shape[0]-1)     # for align corners
        sample_scalew = (nsize[1]-1) / float(self.image_shape[1]-1)
        coords_new = coords_grid(1, *self.image_shape).float()          # 1, 2, H, W
        coordsw = coords_new[:, :1, :, :] * sample_scalew
        coordsh = coords_new[:, 1:, :, :] * sample_scaleh
        coords = torch.cat([coordsw, coordsh], dim=1)
        interp = grid_sampler(flow, coords, mode=mode)                  # 1, 2, H, W

        # gradient filter
        scaleh = nsize[0] / float(self.image_shape[0])
        scalew = nsize[1] / float(self.image_shape[1])
        cw, ch = torch.floor(coords).split(1, dim=1)                    # 1, H, W
        validp = valid[(ch.squeeze().long(), cw.squeeze().long())][None, None, ...]     # 1, 1, H, W
        interp = torch.where(validp.bool(), interp, torch.tensor(np.nan))               # 1, 2, H, W
        interp[:, 0, :, :] /= scalew
        interp[:, 1, :, :] /= scaleh
        
        # outbound filter
        flowed_coord = (coords_new + interp).squeeze(0)
        outbound = (flowed_coord[0] < 0) + (flowed_coord[1] < 0) + \
                    (flowed_coord[0] > self.image_shape[1]-1) + \
                    (flowed_coord[1] > self.image_shape[0]-1)           # H, W
        outbound = outbound.bool()[None, None, ...].repeat(1, 2, 1, 1)  # 1, 2, H, W
        interp[outbound] = torch.tensor(np.nan)

        return interp

    def shape_aug(self, fg, bg, a, wb=None, wf=None, scales=[1.0, 1.25, 1.5, 1.75, 2.0]):
        with torch.no_grad():
            H, W = self.VIDEO_SHAPE
            length = len(fg)
            pfg, pbg, pa = [None] * length, [None] * length, [None] * length
            of_exist = wb is not None and wf is not None
            if of_exist:
                pwb, pwf = [None] * length, [None] * length
            good_sample = False
            while not good_sample:
                scale = random.choice(scales)
                assert self.image_shape[0] == self.image_shape[1]
                nsize = (int(self.image_shape[0] * scale), int(self.image_shape[1] * scale))
                ph = random.randint(0, H - nsize[0] - 1)
                pw = random.randint(0, W - nsize[1] - 1)
                good_sample = True
                for i in range(length):
                    pa[i] = self.img_crop_and_resize(a[i], ph, pw, nsize).squeeze(0)
                    valid = torch.sum((pa[i] > 0) * (pa[i] < 255))
                    if valid.item() < 1:
                        good_sample = False
                        break
            for i in range(length):
                pfg[i] = self.img_crop_and_resize(fg[i], ph, pw, nsize).squeeze(0)
                pbg[i] = self.img_crop_and_resize(bg[i], ph, pw, nsize).squeeze(0)
            if of_exist:
                for i in range(2, length-2):
                    pwb[i] = self.flow_crop_and_resize(wb[i], ph, pw, nsize).squeeze(0)
                    pwf[i] = self.flow_crop_and_resize(wf[i], ph, pw, nsize).squeeze(0)
                pwb[-2] = self.flow_crop_and_resize(wb[-2], ph, pw, nsize).squeeze(0)
                pwf[1] = self.flow_crop_and_resize(wf[1], ph, pw, nsize).squeeze(0)
                for i in range(length):
                    if pwb[i] is None:
                        pwb[i] = torch.ones_like(pwb[length // 2]) * torch.tensor(np.nan)
                    if pwf[i] is None:
                        pwf[i] = torch.ones_like(pwf[length // 2]) * torch.tensor(np.nan)
        if of_exist:
            return pfg, pbg, pa, pwb, pwf
        return pfg, pbg, pa, None, None

    def parse(self, f, length=None):
        if length is None:
            length = self.sample_length
        samples = []
        for v in f:
            v = v.strip()
            fns = [k for k in sorted(self.frame_corr.keys()) if os.path.dirname(k) == v]
            #fns = sorted(os.listdir(os.path.join(self.data_root, self.FG_FOLDER, v)))
            for i in range(len(fns)):
                sample = [None] * length
                c = length // 2
                sample[c] = fns[i]
                for j in range(length // 2):
                    sample[c-j-1] = fns[i-j-1] if i-j-1 >= 0 else fns[-(i-j-1)]
                    sample[c+j+1] = fns[i+j+1] if i+j+1 < len(fns) else fns[len(fns)-(i+j+1)-2]
                samples.append(sample)
        return samples

    def possible_pad(self, t, padvalue=0):
        H, W = t.shape[-2:]
        if H == self.image_shape[0] and W == self.image_shape[1]:
            return t
        assert H <= self.image_shape[0] and W <= self.image_shape[1]
        ph, pw = self.image_shape[0] - H, self.image_shape[1] - W
        if isinstance(padvalue, (int, float)):
            return F.pad(t, (0, pw, 0, ph), value=padvalue)
        elif isinstance(padvalue, (list, tuple)):
            assert len(padvalue) == t.shape[-3]
            mask = F.pad(torch.zeros(H, W), (0, pw, 0, ph), value=1).bool()
            t = F.pad(t, (0, pw, 0, ph), value=0)
            v = torch.tensor(padvalue, dtype=t.dtype).unsqueeze(-1)
            t[:, mask] = v
            return t

    def __getitem__(self, idx):
        def _flow_read(fa, fb):
            if self.FLOW_FOLDER == 'flow_png':
                x = cv2.imread(os.path.join(data_root, self.FLOW_FOLDER, dn, 'flow_{}_{}.png'.format(fa, fb)), cv2.IMREAD_UNCHANGED)
                flow = np.float32(np.int16(x[..., :-1]))
                mask = x[..., -1]
            else:
                flow = np.float32(np.load(os.path.join(data_root, self.FLOW_FOLDER, dn, 'flow_{}_{}.npy'.format(fa, fb))))
                mask = cv2.imread(os.path.join(data_root, self.FLOW_FOLDER, dn, 'flow_{}_{}.png'.format(fa, fb)), cv2.IMREAD_GRAYSCALE)
            invalid = mask == 0
            flow[invalid] = np.nan
            return torch.from_numpy(flow) / self.FLOW_QUANTIZATION_SCALE

        sample = self.samples[idx]
        if self.mode == 'train' and random.random() > 0.5:
            sample = sample[::-1]
        length = len(sample)
        fg, bg, a = [None] * length, [None] * length, [None] * length
        dn = os.path.dirname(sample[0])

        data_root = self.data_root if self.precomputed_val is None else self.precomputed_val

        # img I/O
        for i in range(length):
            _f = cv2.imread(os.path.join(data_root, self.FG_FOLDER, sample[i]), cv2.IMREAD_UNCHANGED)
            bgp = os.path.join(data_root, self.BG_FOLDER, self.frame_corr[sample[i]])
            if not os.path.exists(bgp):
                bgp = os.path.splitext(bgp)[0]+'.png'
            bg[i] = np.float32(cv2.imread(bgp, cv2.IMREAD_COLOR))
            fg[i] = np.float32(_f[..., :-1])
            a[i] = np.float32(_f[..., -1:])
            assert bg[i].shape[:2] == fg[i].shape[:2]

        # optical flow I/O
        if not self.no_flow:
            wb, wf = [None] * length, [None] * length
            fns = []
            for i in range(length):
                fns.append(os.path.splitext(os.path.basename(sample[i]))[0])
            for i in range(2, length-2):
                wf[i] = _flow_read(fns[i], fns[i+1])
                wb[i] = _flow_read(fns[i], fns[i-1])
            wf[1] = _flow_read(fns[1], fns[2])
            wb[-2] = _flow_read(fns[-2], fns[-3])
        else:
            wf = None
            wb = None
            
        # augmentation
        if self.mode == 'train':
            fg_aug = self.pixel_aug.to_deterministic()
            bg_aug = self.pixel_aug.to_deterministic()
            jpeg_aug = self.jpeg_aug.to_deterministic()
            fg, bg, a, wb, wf = self.shape_aug(fg, bg, a, wb, wf)
            for i in range(length):
                fg[i] = fg_aug.augment_image(np.uint8(fg[i].permute(1, 2, 0).numpy()))
                fg[i] = jpeg_aug.augment_image(fg[i])
                fg[i] = torch.from_numpy(fg[i]).permute(2, 0, 1).float()
                bg[i] = bg_aug.augment_image(np.uint8(bg[i].permute(1, 2, 0).numpy()))
                bg[i] = torch.from_numpy(bg[i]).permute(2, 0, 1).float()
        else:
            if self.precomputed_val is not None:
                img_padding_value = [103.53, 116.28, 123.675] # BGR
                for i in range(length):
                    fg[i] = self.possible_pad(torch.from_numpy(fg[i]).permute(2, 0, 1), img_padding_value)
                    bg[i] = self.possible_pad(torch.from_numpy(bg[i]).permute(2, 0, 1), img_padding_value)
                    a[i] = self.possible_pad(torch.from_numpy(a[i]).permute(2, 0, 1))
                if not self.no_flow:
                    for i in range(2, length-2):
                        wb[i] = self.possible_pad(wb[i].permute(2, 0, 1), np.nan)
                        wf[i] = self.possible_pad(wf[i].permute(2, 0, 1), np.nan)
                    wb[-2] = self.possible_pad(wb[-2].permute(2, 0, 1), np.nan)
                    wf[1] = self.possible_pad(wf[1].permute(2, 0, 1), np.nan)
            else:
                for i in range(length):
                    fg[i] = self.img_crop_and_resize(fg[i], 0, 0).squeeze(0)
                    bg[i] = self.img_crop_and_resize(bg[i], 0, 0).squeeze(0)
                    a[i] = self.img_crop_and_resize(a[i], 0, 0).squeeze(0)
                if not self.no_flow:
                    for i in range(2, length-2):
                        wb[i] = self.flow_crop_and_resize(wb[i], 0, 0).squeeze(0)
                        wf[i] = self.flow_crop_and_resize(wf[i], 0, 0).squeeze(0)
                    wb[-2] = self.flow_crop_and_resize(wb[-2], 0, 0).squeeze(0)
                    wf[1] = self.flow_crop_and_resize(wf[1], 0, 0).squeeze(0)
            if not self.no_flow:
                for i in range(length):
                    if wb[i] is None:
                        wb[i] = torch.ones_like(wb[length // 2]) * torch.tensor(np.nan)
                    if wf[i] is None:
                        wf[i] = torch.ones_like(wf[length // 2]) * torch.tensor(np.nan)

        fg = torch.stack(fg).float()
        bg = torch.stack(bg).float()
        a = torch.stack(a).float()
        if not self.no_flow:
            wb = torch.stack(wb).float()
            wf = torch.stack(wf).float()
            # Everything here is [S, N, H, W]
            return fg, bg, a, wb, wf, torch.tensor(idx)
        return fg, bg, a, torch.tensor(idx)