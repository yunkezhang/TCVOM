import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import models.VMN as VMN
import utils.loss_func as L
from models.DIM.vggnet import DIM_VGG
from models.GCA.generators import GCA
from models.FBA.models import FBA
from models.Index.net import IndexMatting
from utils.utils import \
    coords_grid, flow_dt, grid_sampler, \
    trimap_transform

class FullModel(nn.Module):
    ARCH_DICT = {
        'gca': GCA,
        'dim': DIM_VGG,
        'fba': FBA,
        'index': IndexMatting
    }
    TRIMAP_CHANNEL_DICT = {
        'gca': 3,
        'dim': 1,
        'index': 1,
        'fba': 8,
    }
    FBA_LOSS_NORMALIZE = True
    FBA_L_ATT_MULTIPLIER = 1

    def __init__(self, model, dilate_kernel=None, eps=0, **kwargs):
        super(FullModel, self).__init__()
        self.DILATION_KERNEL = dilate_kernel
        self.EPS = eps
        self.IMG_SCALE = 1./255
        self.register_buffer('IMG_MEAN', torch.tensor([0.485, 0.456, 0.406]).reshape([1, 1, 3, 1, 1]).float())
        self.register_buffer('IMG_STD', torch.tensor([0.229, 0.224, 0.225]).reshape([1, 1, 3, 1, 1]).float())
        self.model_name = model
        if model.startswith('vmn'):
            # ***************************** #
            #           VMN ARCH            #
            # ***************************** #
            self.NET = VMN.get_VMN_models(arch=model, **kwargs)
            self.window = kwargs['agg_window']
        else:
            ARCH = self.ARCH_DICT[model]
            self.NET = ARCH()

        self.method = model[model.rfind('_')+1:]
        if self.method == 'fba':
            self.LAPLOSS = L.LapLoss()
        self.TRIMAP_CHANNEL = self.TRIMAP_CHANNEL_DICT[self.method]

    def make_trimap(self, alpha):
        b = alpha.shape[0]
        alpha = torch.where(alpha < self.EPS, torch.zeros_like(alpha), alpha)
        alpha = torch.where(alpha > 1 - self.EPS, torch.ones_like(alpha), alpha)
        trimasks = ((alpha > 0) & (alpha < 1.)).float().split(1)    # trimasks: B * [S, C, H, W]
        trimaps = [None] * b
        for i in range(b):
            # trimap width: 1 - 51
            kernel_rad = int(torch.randint(0, 26, size=())) \
                if self.DILATION_KERNEL is None else self.DILATION_KERNEL
            trimaps[i] = F.max_pool2d(trimasks[i].squeeze(0), kernel_size=kernel_rad*2+1, stride=1, padding=kernel_rad)
        trimap = torch.stack(trimaps)
        if self.TRIMAP_CHANNEL == 1:
            trimap1 = torch.where(trimap > 0.5, 128.*torch.ones_like(alpha)*self.IMG_SCALE, alpha)
            return trimap1, trimap
        elif self.TRIMAP_CHANNEL == 3:
            # 0: bg, 1: un, 2: fg
            trimap1 = torch.where(trimap > 0.5, torch.ones_like(alpha), 2 * alpha).long()
            trimap3 = F.one_hot(trimap1.squeeze(2), num_classes=3).permute(0, 1, 4, 2, 3)
            return trimap3.float(), trimap
        elif self.TRIMAP_CHANNEL == 8:
            trimap1 = torch.where(trimap > 0.5, 255*torch.ones_like(alpha), alpha)
            trimap2f = (trimap1 == 1).float()
            trimap2b = (trimap1 == 0).float()
            trimap2 = torch.cat([trimap2b, trimap2f], dim=2)    # [B, S, 2, H, W]
            transformed_trimap = trimap_transform(trimap2)      # [B, S, 6, H, W]
            return torch.cat([transformed_trimap, trimap2], dim=2).float(), trimap

    def preprocess(self, a, fg, bg):
        # Data preprocess
        with torch.no_grad():
            scaled_gts = a * self.IMG_SCALE
            scaled_fgs = fg.flip([2]) * self.IMG_SCALE
            scaled_bgs = bg.flip([2]) * self.IMG_SCALE
            scaled_imgs = scaled_fgs * scaled_gts + scaled_bgs * (1. - scaled_gts)
            scaled_tris, trimasks = self.make_trimap(scaled_gts)
            #alphas, features = [None] * self.sample_length, [None] * self.sample_length
            imgs = ((scaled_imgs - self.IMG_MEAN) / self.IMG_STD)#.split(1, dim=1)
        return scaled_imgs, scaled_fgs, scaled_bgs, scaled_gts, scaled_tris, trimasks, imgs

    def single_image_loss(self, preds, trimasks, \
        scaled_gts, scaled_fgs, scaled_bgs, scaled_imgs, start, end):
        L_alpha, L_comp, L_grad = [], [], []
        sample_length = preds.shape[1]
        alphas, comps = [None] * sample_length, [None] * sample_length
        for c in range(start, end):
            c_gt = scaled_gts[:, c, ...]
            c_trimask = trimasks[:, c, ...].float()
            c_refine = torch.where(c_trimask.bool(), preds[:, c, ...], c_gt)
            alphas[c] = c_refine
            
            c_comp = scaled_fgs[:, c, ...] * c_refine + scaled_bgs[:, c, ...] * (1. - c_refine)
            c_img = scaled_imgs[:, c, ...]
            comps[c] = c_comp
            L_alpha.append(L.L1_mask(c_refine, c_gt, c_trimask))
            if self.method != 'gca': # GCA only uses alpha loss
                L_comp.append(L.L1_mask(c_comp, c_img, c_trimask))
                L_grad.append(L.L1_grad(c_refine, c_gt, c_trimask))
        if self.method == 'gca':
            L_comp = torch.zeros_like(L_alpha[0])
            L_grad = torch.zeros_like(L_alpha[0])
        else:
            L_comp = sum(L_comp) / float(len(L_comp))
            L_grad = sum(L_grad) / float(len(L_grad))
        L_alpha = sum(L_alpha) / float(len(L_alpha))
        for i in range(start):
            comps[i] = torch.zeros_like(comps[start])
            comps[-i-1] = torch.zeros_like(comps[start])
            alphas[i] = torch.zeros_like(alphas[start])
            alphas[-i-1] = torch.zeros_like(alphas[start])
        comps = torch.stack(comps, dim=1).clamp(0, 1)
        alphas = torch.stack(alphas, dim=1).clamp(0, 1)

        return L_alpha, L_comp, L_grad, alphas, comps

    def fba_single_image_loss(self, preds, trimasks, \
        scaled_gts, scaled_fgs, scaled_bgs, scaled_imgs, start, end,
        normalize):
        # Since FBA also outputs F and B...
        # preds [B, S, 7, H, W]
        sample_length = preds.shape[1]
        alpha = preds[:, :, :1, ...]
        predF = preds[:, :, 1:4, ...]
        predB = preds[:, :, 4:, ...]
        L_alpha_comp, L_lap, L_grad = [], [], []
        alphas, comps, Fs, Bs = [None] * sample_length, [None] * sample_length, \
                                [None] * sample_length, [None] * sample_length
        
        for c in range(start, end):
            c_gt = scaled_gts[:, c, ...]
            c_trimask = trimasks[:, c, ...]
            c_refine = torch.where(c_trimask.bool(), alpha[:, c, ...], c_gt)
            c_img = scaled_imgs[:, c, ...]
            #c_fgmask = ((c_trimask == 1) + (c_gt == 1)).bool().repeat(1, 3, 1, 1)
            #c_bgmask = ((c_trimask == 1) + (c_gt == 0)).bool().repeat(1, 3, 1, 1)
            c_F = torch.where(c_trimask.bool().repeat(1, 3, 1, 1), \
                              predF[:, c, ...], scaled_fgs[:, c, ...])
            c_B = torch.where(c_trimask.bool().repeat(1, 3, 1, 1), \
                              predB[:, c, ...], scaled_bgs[:, c, ...])
            alphas[c] = c_refine
            comps[c] = c_F * c_refine + c_B * (1. - c_refine)
            Fs[c] = c_F
            Bs[c] = c_B
            
            # There's no mean op in FBA paper, so we'll only sum (normalize=False)
            # L1 and comp related losses
            L_a1 = L.L1_mask(c_refine, c_gt, normalize=normalize)
            ac = c_F * c_gt + c_B * (1. - c_gt)
            L_ac = L.L1_mask(ac, c_img, normalize=normalize)
            FBc = scaled_fgs[:, c, ...] * c_refine + scaled_bgs[:, c, ...] * (1. - c_refine)
            L_FBc = L.L1_mask(FBc, c_img, normalize=normalize)
            L_FB1 = L.L1_mask(c_F, scaled_fgs[:, c, ...], normalize=normalize) + \
                    L.L1_mask(c_B, scaled_bgs[:, c, ...], normalize=normalize)
            L_alpha_comp.append(L_a1 + L_ac + 0.25 * (L_FBc + L_FB1))
            
            # gradient related losses
            L_ag = L.L1_grad(c_refine, c_gt, normalize=normalize)
            #L_grad.append(L_ag)
            L_FBexcl = L.exclusion_loss(c_F, c_B, level=3, normalize=normalize)
            L_grad.append(L_ag + 0.25 * L_FBexcl)

            # Laplacian loss
            L_a_lap = self.LAPLOSS(c_refine, c_gt, normalize=normalize)
            L_F_lap = self.LAPLOSS(c_F, scaled_fgs[:, c, ...], normalize=normalize)
            L_B_lap = self.LAPLOSS(c_B, scaled_bgs[:, c, ...], normalize=normalize)
            L_lap.append(L_a_lap + 0.25 * (L_F_lap + L_B_lap))
        L_alpha_comp = sum(L_alpha_comp) / float(len(L_alpha_comp))
        L_grad = sum(L_grad) / float(len(L_grad))
        L_lap = sum(L_lap) / float(len(L_lap))

        for i in range(start):
            comps[i] = torch.zeros_like(comps[start])
            comps[-i-1] = torch.zeros_like(comps[start])
            alphas[i] = torch.zeros_like(alphas[start])
            alphas[-i-1] = torch.zeros_like(alphas[start])
            Fs[i] = torch.zeros_like(Fs[start])
            Fs[-i-1] = torch.zeros_like(Fs[start])
            Bs[i] = torch.zeros_like(Bs[start])
            Bs[-i-1] = torch.zeros_like(Bs[start])
        comps = torch.stack(comps, dim=1)
        Fs = torch.stack(Fs, dim=1)
        Bs = torch.stack(Bs, dim=1)
        alphas = torch.stack(alphas, dim=1)
        return L_alpha_comp, L_lap, L_grad, alphas, comps, Fs, Bs

    def forward(self, a, fg, bg):
        sample_length = a.shape[1]
        c = sample_length // 2
        scaled_imgs, scaled_fgs, scaled_bgs, scaled_gts, tris, trimasks, imgs = self.preprocess(a, fg, bg)

        inputs = list(torch.cat([imgs, tris], dim=2).split(1, dim=1))
        if not self.model_name.startswith('vmn'):
            extras = None if self.method != 'fba' else \
                [scaled_imgs[:, c, ...], tris[:, c, -2:, ...]]
            preds = [None] * sample_length
            preds[c] = self.NET(inputs[c].squeeze(1), extras=extras)
            for i in range(c):
                preds[i] = torch.zeros_like(preds[c])
                preds[-i-1] = torch.zeros_like(preds[c])
            start, end = c, c+1
        else:
            masks = trimasks.split(1, dim=1)
            extras = None if self.method != 'fba' else \
                [[scaled_imgs[:, i], tris[:, i, -2:]] for i in range(sample_length)]
            preds = self.NET(inputs, masks, extras=extras)[0]
            start, end = 1, sample_length-1
        preds = torch.stack(preds, dim=1)

        # Single image loss
        loss_inputs = (preds, trimasks, scaled_gts, \
            scaled_fgs, scaled_bgs, scaled_imgs, \
            start, end)
        if self.method != 'fba':
            # L_alpha, L_comp, L_grad
            loss1, loss2, loss3, alphas, comps = \
            self.single_image_loss(*loss_inputs)
            Fs = scaled_fgs
            Bs = scaled_bgs
        else:
            # L_alpha_comp, L_lap, L_grad
            loss1, loss2, loss3, alphas, comps, Fs, Bs = \
                self.fba_single_image_loss(*loss_inputs, normalize=self.FBA_LOSS_NORMALIZE)

        with torch.no_grad():
            if self.TRIMAP_CHANNEL != 1:
                tris_vis = torch.where(trimasks.bool(), \
                    torch.ones_like(scaled_gts)*128*self.IMG_SCALE, \
                    scaled_gts)
            else:
                tris_vis = tris
        return [loss1, loss2, loss3,                # Loss
                scaled_imgs, tris_vis, alphas, comps,   # Vis
                scaled_gts, Fs, Bs]

class FullModel_VMD(FullModel):
    TAM_OS = 8
    def __init__(self, model, att_thres=0.3, label_smooth=0.2, **kwargs):
        assert model.startswith('vmn'), "FullModel_VMD only support VMN arch"
        super(FullModel_VMD, self).__init__(model, **kwargs)
        self.att_thres = att_thres
        self.label_smooth = label_smooth
        self.AttBCE = torch.nn.BCEWithLogitsLoss(reduction='mean' \
            if self.method == 'fba' and not self.FBA_LOSS_NORMALIZE else 'mean')

    def forward(self, a, fg, bg, wb=None, wf=None):
        batch_size, sample_length = a.shape[:2]
        scaled_imgs, scaled_fgs, scaled_bgs, scaled_gts, tris, trimasks, imgs = self.preprocess(a, fg, bg)

        inputs = list(torch.cat([imgs, tris], dim=2).split(1, dim=1))
        extras = None if self.method != 'fba' else \
            [[scaled_imgs[:, i], tris[:, i, -2:]] for i in range(sample_length)]
        masks = trimasks.split(1, dim=1)
        preds, attb, attf, small_mask = self.NET(inputs, masks, extras=extras)
        preds = torch.stack(preds, dim=1)

        # Single image loss
        loss_inputs = (preds, trimasks, scaled_gts, \
            scaled_fgs, scaled_bgs, scaled_imgs, \
            1, sample_length-1)
        if self.method != 'fba':
            # L_alpha, L_comp, L_grad
            loss1, loss2, loss3, alphas, comps = \
                self.single_image_loss(*loss_inputs)
            Fs = scaled_fgs
            Bs = scaled_bgs
        else:
            # L_alpha_comp, L_lap, L_grad
            loss1, loss2, loss3, alphas, comps, Fs, Bs = \
                self.fba_single_image_loss(*loss_inputs, \
                    normalize=self.FBA_LOSS_NORMALIZE)

        # Attention map loss
        L_att = []
        H, W = scaled_gts.shape[-2:]
        H = H // self.TAM_OS
        W = W // self.TAM_OS
        for c in range(1, sample_length-1):
            bgt = F.avg_pool2d(scaled_gts[:, c-1, ...], self.TAM_OS, stride=self.TAM_OS)    # B, 1, H, W
            fgt = F.avg_pool2d(scaled_gts[:, c+1, ...], self.TAM_OS, stride=self.TAM_OS)    # B, 1, H, W
            cgt = F.avg_pool2d(scaled_gts[:, c, ...], self.TAM_OS, stride=self.TAM_OS)      # B, 1, H, W
            m = small_mask[c].reshape(batch_size, -1)   # B, 1, H, W -> B, HW
            # BCEWithLogitsLoss will return nan if there's no valid pixel
            if m.float().sum() == 0:
                L_att.append(torch.zeros_like(loss1))
                continue
            b = attb[c].reshape(batch_size, -1, H*W).permute(1, 0, 2)   # w**2, B, HW
            f = attf[c].reshape(batch_size, -1, H*W).permute(1, 0, 2)   # w**2, B, HW
            cb = b[:, m]    # [w**2, BU]
            cf = f[:, m]    # [w**2, BU]

            # construct groundtruth
            with torch.no_grad():
                bgt_unfold = F.unfold(bgt, kernel_size=self.window, \
                    padding=self.window // 2).reshape(batch_size, -1, H*W).permute(1, 0, 2)  # w**2, B, HW
                fgt_unfold = F.unfold(fgt, kernel_size=self.window, \
                    padding=self.window // 2).reshape(batch_size, -1, H*W).permute(1, 0, 2)  # w**2, B, HW
                cgt = cgt.reshape(batch_size, 1, H*W).permute(1, 0, 2)                       # 1, B, HW
                bgt_unfold = bgt_unfold[:, m]   # w**2, BU
                fgt_unfold = fgt_unfold[:, m]   # w**2, BU
                cgt = cgt[:, m]                 # 1, BU

                dcb = torch.abs(cgt - bgt_unfold)       # w**2, BU
                dcb = (dcb < self.att_thres).float() * (1 - self.label_smooth)
                dcf = torch.abs(cgt - fgt_unfold)       # w**2, BU
                dcf = (dcf < self.att_thres).float() * (1 - self.label_smooth)
            loss = self.AttBCE(cb, dcb) + self.AttBCE(cf, dcf)
            L_att.append(loss / 2.0)
        L_att = sum(L_att) / float(len(L_att))
        if self.method == 'fba':
            L_att = L_att * self.FBA_L_ATT_MULTIPLIER

        # Temp loss: dtSSD
        def _dtSSD(pred, gt, normalize=True):
            L_dt = []
            for c in range(1, sample_length-2):
                dadt = pred[:, c, ...] - pred[:, c+1, ...]
                dgtdt = gt[:, c, ...] - gt[:, c+1, ...]
                L_dt.append(L.L1_mask(dadt, dgtdt, trimasks[:, c, ...], normalize=normalize))
            L_dt = sum(L_dt) / float(len(L_dt))
            return L_dt

        if sample_length >= 5:
            if self.method == 'fba':
                # for FBA we also supervise Fs and Bs
                L_dt = _dtSSD(alphas, scaled_gts, normalize=self.FBA_LOSS_NORMALIZE)
                L_F_dt = _dtSSD(Fs, scaled_fgs, normalize=self.FBA_LOSS_NORMALIZE)
                L_B_dt = _dtSSD(Bs, scaled_bgs, normalize=self.FBA_LOSS_NORMALIZE)
                L_dt = L_dt + 0.25 * (L_F_dt + L_B_dt)
            else:
                L_dt = _dtSSD(alphas, scaled_gts)
        else:
            L_dt = torch.zeros_like(L_att)

        with torch.no_grad():
            if self.TRIMAP_CHANNEL != 1:
                tris_vis = torch.where(trimasks.bool(), \
                    torch.ones_like(scaled_gts)*128*self.IMG_SCALE, \
                    scaled_gts)
            else:
                tris_vis = tris

        return [loss1, loss2, loss3, L_dt, L_att,   # Loss
                scaled_imgs, tris_vis, alphas, comps,       # Vis
                scaled_gts, Fs, Bs]

class EvalModel(FullModel):
    def preprocess(self, img, tri):
        # Data preprocess
        with torch.no_grad():
            #scaled_gts = a * self.IMG_SCALE
            #scaled_fgs = fg.flip([2]) * self.IMG_SCALE
            #scaled_bgs = bg.flip([2]) * self.IMG_SCALE
            scaled_imgs = img.float().flip([2]) * self.IMG_SCALE
            imgs = ((scaled_imgs - self.IMG_MEAN) / self.IMG_STD)
            scaled_tris = tri.float() * self.IMG_SCALE
            trimask = ((scaled_tris > 0) & (scaled_tris < 1))
            if self.DILATION_KERNEL is not None:
                trimask = trimask.float().split(1)    # trimasks: B * [S, C, H, W]
                b = len(trimask)
                trimasks = [None] * b
                for i in range(b):
                    trimasks[i] = F.max_pool2d(trimask[i].squeeze(0), \
                        kernel_size=self.DILATION_KERNEL*2+1, stride=1, padding=self.DILATION_KERNEL)
                trimask = torch.stack(trimasks).bool()
            if self.TRIMAP_CHANNEL == 3:
                trimap1 = torch.where(trimask, torch.ones_like(scaled_tris), 2 * scaled_tris).long()
                scaled_tris = F.one_hot(trimap1.squeeze(2), num_classes=3).permute(0, 1, 4, 2, 3)
            elif self.TRIMAP_CHANNEL == 8:
                trimap2f = (scaled_tris == 1).float()
                trimap2b = (scaled_tris == 0).float()
                trimap2 = torch.cat([trimap2b, trimap2f], dim=2)    # [B, S, 2, H, W]
                transformed_trimap = trimap_transform(trimap2)      # [B, S, 6, H, W]
                scaled_tris = torch.cat([transformed_trimap, trimap2], dim=2).float()
        return scaled_imgs, scaled_tris, trimask.float(), imgs

    def forward(self, imgs, tris):
        sample_length = imgs.shape[1]
        c = sample_length // 2
        scaled_imgs, scaled_tris, trimasks, imgs = self.preprocess(imgs, tris)

        inputs = list(torch.cat([imgs, scaled_tris], dim=2).split(1, dim=1))
        if not self.model_name.startswith('vmn'):
            preds = [None] * sample_length
            extras = None if self.method != 'fba' else \
                [scaled_imgs[:, c, ...], scaled_tris[:, c, -2:, ...]]
            preds[c] = self.NET(inputs[c].squeeze(1), extras=extras)
            for i in range(c):
                preds[i] = torch.zeros_like(preds[c])
                preds[-i-1] = torch.zeros_like(preds[c])
            start, end = c, c+1
        else:
            masks = trimasks.split(1, dim=1)
            extras = None if self.method != 'fba' else \
                [[scaled_imgs[:, i], scaled_tris[:, i, -2:]] for i in range(sample_length)]
            preds, attb, attf, small_mask = self.NET(inputs, masks, extras=extras)
            start, end = 1, sample_length-1
        preds = torch.stack(preds, dim=1)

        # Loss & Vis
        if self.method != 'fba':
            alphas = [None] * sample_length
            for c in range(start, end):
                c_gt = tris[:, c, ...].float() * self.IMG_SCALE
                c_trimask = trimasks[:, c, ...]
                c_refine = torch.where(c_trimask.bool(), preds[:, c, ...], c_gt)
                alphas[c] = c_refine
            for i in range(start):
                alphas[i] = torch.zeros_like(alphas[start])
                alphas[-i-1] = torch.zeros_like(alphas[start])
            alphas = torch.stack(alphas, dim=1)
            return alphas
        else:
            alpha = preds[:, :, :1, ...]
            predF = preds[:, :, 1:4, ...]
            predB = preds[:, :, 4:, ...]
            alphas, Fs, Bs = [None] * sample_length, \
                             [None] * sample_length, [None] * sample_length
            for c in range(start, end):
                c_gt = tris[:, c, ...].float() * self.IMG_SCALE
                c_trimask = trimasks[:, c, ...]
                c_refine = torch.where(c_trimask.bool(), alpha[:, c, ...], c_gt)
                c_img = scaled_imgs[:, c, ...]
                c_F = torch.where(c_trimask.bool().repeat(1, 3, 1, 1), \
                                predF[:, c, ...], scaled_imgs[:, c, ...])
                c_B = torch.where(c_trimask.bool().repeat(1, 3, 1, 1), \
                                predB[:, c, ...], scaled_imgs[:, c, ...])
                alphas[c] = c_refine
                Fs[c] = c_F
                Bs[c] = c_B
            for i in range(start):
                alphas[i] = torch.zeros_like(alphas[start])
                alphas[-i-1] = torch.zeros_like(alphas[start])
                Fs[i] = torch.zeros_like(Fs[start])
                Fs[-i-1] = torch.zeros_like(Fs[start])
                Bs[i] = torch.zeros_like(Bs[start])
                Bs[-i-1] = torch.zeros_like(Bs[start])
            alphas = torch.stack(alphas, dim=1)
            Fs = torch.stack(Fs, dim=1)
            Bs = torch.stack(Bs, dim=1)
            return alphas, Fs, Bs
