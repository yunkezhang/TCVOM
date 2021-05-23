import logging
import os
import pickle
import random
import time

import cv2
import imgaug
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from imgaug import parameters as iap


class DIMPretrainDataset(torch.utils.data.Dataset):

    def __init__(self, data_root, image_shape=None, min_shape=None, val_mode=None, plus1=True, isTrain=True):
        '''
        We assume that image_shape is always an even number.
        With plus1 = True, we'll modify image_shape to odd.
        '''
        self.data_root = data_root
        if isTrain:
            assert image_shape is not None
            self.image_shape = image_shape
        else:
            assert min_shape is not None
            self.image_shape = (min_shape, min_shape)
        self.isTrain = isTrain
        self.val_mode = val_mode
        if plus1:
            self.image_shape = (image_shape[0]+1, image_shape[1]+1)
        if isTrain:
            bg_set_fn = [i.strip() for i in open(os.path.join(data_root, 'bg_train_set.txt'), 'r')]
            fg_set = [i.strip() for i in open(os.path.join(data_root, 'fg_train_set_old.txt'), 'r')]
            fg_set_fn = [None] * len(bg_set_fn)
            for i in range(len(bg_set_fn)):
                fg_set_fn[i] = fg_set[i // 100]
        else:
            bg_set_fn = [i.strip() for i in open(os.path.join(data_root, 'bg_val_set.txt'), 'r')]#[60:]
            fg_set_fn = [i.strip() for i in open(os.path.join(data_root, 'fg_val_set.txt'), 'r')]#[60:]

        self.dataset_length = len(bg_set_fn)
        assert len(fg_set_fn) == self.dataset_length
        #ratio = self.dataset_length // len(fg_set_fn)
        self.sample_fn = [None] * self.dataset_length
        for i in range(self.dataset_length):
            _fn = fg_set_fn[i].split(' ')
            self.sample_fn[i] = (_fn[0], _fn[1], bg_set_fn[i])

        # apply to fg, bg
        self.flip_and_color_aug = iaa.Sequential([
            iaa.MultiplyHueAndSaturation(mul=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5)), # mean, std, low, high
            iaa.GammaContrast(gamma=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5)),
            iaa.AddToHue(value=iap.TruncatedNormal(0.0, 0.1*100, -0.2*255, 0.2*255))
        ])

        if isTrain:
            self.min_shape = min_shape
            self.preshape_aug = iaa.Sequential([
                iaa.CropToFixedSize(width=self.min_shape, height=self.min_shape, \
                    position='uniform' if isTrain else 'center'),
            ])
        else:
            assert self.val_mode in ['gca', 'dim', 'origin', 'resize']
            if self.val_mode == 'resize':
                assert min_shape is not None
                self.min_shape = min_shape
            elif self.val_mode == 'origin':
                print ('Warning: val_mode == origin, change min_shape to 2112')
                self.min_shape = 2112
                self.image_shape = (2112, 2112)

        self.shape_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.OneOf([
                # iaa.CropToFixedSize(width=384, height=384),
                # iaa.CropToFixedSize(width=448, height=448),
                iaa.CropToFixedSize(width=512, height=512),
                iaa.CropToFixedSize(width=576, height=576),
                iaa.CropToFixedSize(width=640, height=640),
                iaa.CropToFixedSize(width=704, height=704),
                iaa.CropToFixedSize(width=768, height=768)
            ]),
        ]) if isTrain else \
            iaa.Sequential([
                iaa.CropToFixedSize(width=self.image_shape[1], \
                    height=self.image_shape[0], \
                    position='uniform' if isTrain else 'center'),
            ])

    def __len__(self):
        return self.dataset_length

    def make_frames3(self, fg, bg, alpha):
        # input: fg, bg, alpha
        # output: listfg, listbg, lista
        def _rotate_(M, center = None, angle = None, scale = None):
            M_now = cv2.getRotationMatrix2D((center[0],center[1]), angle, scale)
            M_now = np.concatenate((M_now,[[0,0,1]]))
            return np.matmul(M, M_now)

        def _move_(M,vec):
            M_now = np.float32([[1,0,vec[0]],[0,1,vec[1]]])
            M_now = np.concatenate((M_now,[[0,0,1]]))
            return np.matmul(M,M_now)

        def _get_random_var_(w,h,
                            MOVE_MAX = 20,
                            ROTATE_MAX_CENTER = 10,
                            ROTATE_MAX_ANGLE = 2,
                            ROTATE_MIN_SCALE = 1.00,
                            ROTATE_MAX_SCALE = 1.00):
            center = ROTATE_MAX_CENTER*2*(np.random.random(2)-0.5)+np.array([w/2,h/2],np.float32)
            move = np.random.randint(-MOVE_MAX,MOVE_MAX,[2])
            angle = (np.random.random()-0.5)*2.*ROTATE_MAX_ANGLE
            scale = np.random.random()*(ROTATE_MAX_SCALE-ROTATE_MIN_SCALE)+ROTATE_MIN_SCALE
            #print("center move angle scale")
            #print(center,move,angle,scale)
            return center,move,angle,scale

        def _get_new_M(M, var):
            center, move, angle, scale = var
            _rM = _rotate_(M, center, angle, scale)
            _mM = _move_(_rM,move)
            return _mM

        fgs, bgs, alphas = [None] * 3, [None] * 3, [None] * 3
        good_sample = False

        bh, bw = bg.shape[:2]
        fh, fw = fg.shape[:2]
        dh, dw = (bh - fh) / 2., (bw - fw) / 2.
        rh, rw = (np.random.random() - 0.5) * 2., (np.random.random() - 0.5) * 2.
        I = np.eye(3, dtype=np.float32)
        FM0 = _move_(I, [rh*dh, rw*dw])
        FF_var = _get_random_var_(fw, fh, MOVE_MAX=200, ROTATE_MAX_ANGLE=10, ROTATE_MIN_SCALE=0.9, ROTATE_MAX_SCALE=1.1)
        FSTEP_var = _get_random_var_(fw, fh)
        BSTEP_var = _get_random_var_(bw, bh, ROTATE_MAX_CENTER=0, ROTATE_MAX_ANGLE=0)
        FMs = _get_new_M(I, FSTEP_var)
        BMs = _get_new_M(I, BSTEP_var)

        good_sample = True
        FM0 = np.matmul(_get_new_M(I, FF_var), FM0)
        FM_ = [np.linalg.inv(FMs), I, FMs]
        BM_ = [np.linalg.inv(BMs), I, BMs]
        for i in range(3):
            FM = np.matmul(FM_[i], FM0)
            BM = BM_[i]
            fgs[i] = cv2.warpPerspective(fg, FM, (bw, bh))
            bgs[i] = cv2.warpPerspective(bg, BM, (bw, bh))
            alphas[i] = cv2.warpPerspective(alpha, FM, (bw, bh))[..., np.newaxis]
            if np.sum(np.logical_and(alphas[i] > 0, alphas[i] < 255)) < 400:
                good_sample = False
                
        return fgs, bgs, alphas, good_sample

    def compose(self, fg, bg, a):
        fg = fg / 255.0
        bg = bg / 255.0
        a = a / 255.0
        comp = fg * a + bg * (1. - a)
        return comp.clip(0., 1.) * 255.

    def make_trimap(self, a, size):
        e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        if self.isTrain:
            a = np.where(a < 10, 0, a)
            a = np.where(a > 245, 255, a)
        trimask = np.uint8((a > 0) * (a < 255))
        trimask = cv2.dilate(trimask, e)[..., np.newaxis]
        trimap = np.where(trimask, 128, a)
        return trimap

    def resize_ratio(self, img, a=None, use_short=True, force_resize=False, interpolation=cv2.INTER_AREA):
        tgt = self.min_shape
        short = np.min(img.shape[:2]) if use_short else np.max(img.shape[:2])
        if short < tgt or force_resize:
            ratio = tgt / float(short)
            new_size = (max(int(img.shape[1]*ratio), tgt), max(int(img.shape[0]*ratio), tgt)) \
                if use_short else (min(int(img.shape[1]*ratio), tgt), min(int(img.shape[0]*ratio), tgt))
            img = np.float32(img)
            img = cv2.resize(img, new_size, interpolation=interpolation)
            if a is not None:
                a = np.float32(a)
                a = cv2.resize(a, new_size, interpolation=interpolation)
            if interpolation == cv2.INTER_CUBIC:
                img = np.clip(img, 0, 255)
                if a is not None:
                    a = np.clip(a, 0, 255)
        if a is not None:
            return np.uint8(img), np.uint8(a)
        else:
            return np.uint8(img)

    def resize_as(self, img, tgt, interpolation=cv2.INTER_AREA):
        ratio = max(tgt.shape[0] / float(img.shape[0]), tgt.shape[1] / float(img.shape[1]))
        new_size = (int(np.ceil(img.shape[1]*ratio)), int(np.ceil(img.shape[0]*ratio)))
        img = np.float32(img)
        intp = cv2.resize(img, new_size, interpolation)
        if interpolation == cv2.INTER_CUBIC:
            intp = np.clip(intp, 0, 255)
        return np.uint8(intp)

    def resize_32(self, img, interpolation=cv2.INTER_AREA):
        img = np.float32(img)
        h = int(np.ceil(img.shape[0] / 32.0) * 32)
        w = int(np.ceil(img.shape[1] / 32.0) * 32)
        intp = cv2.resize(img, (w, h), interpolation=interpolation)
        if interpolation == cv2.INTER_CUBIC:
            intp = np.clip(intp, 0, 255)
        return np.uint8(intp)

    def __getitem__(self, idx):
        good_sample = False
        #p_aug = self.flip_and_color_aug.to_deterministic()
        #fg = cv2.imread(os.path.join(self.data_root, self.fg_prefix+'fg', self.sample_fn[idx][0]), cv2.IMREAD_COLOR)
        ofg = cv2.imread(os.path.join(self.data_root, self.sample_fn[idx][0]), cv2.IMREAD_COLOR)
        #a = cv2.imread(os.path.join(self.data_root, self.fg_prefix+'mask', self.sample_fn[idx][0]), cv2.IMREAD_GRAYSCALE)
        oa = cv2.imread(os.path.join(self.data_root, self.sample_fn[idx][1]), cv2.IMREAD_GRAYSCALE)
        obg = cv2.imread(os.path.join(self.data_root, self.sample_fn[idx][2]), cv2.IMREAD_COLOR)
        og_shape = ofg.shape[:2]
        #fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
        #bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        while not good_sample:
            fg = np.array(ofg)
            bg = np.array(obg)
            a = np.array(oa)
            if self.isTrain:
                pre_aug = self.preshape_aug.to_deterministic()
                fg, a = self.resize_ratio(fg, a)
                bg = self.resize_ratio(bg, interpolation=cv2.INTER_CUBIC)
                
                fg = pre_aug.augment_image(fg)
                bg = self.preshape_aug.augment_image(bg)
                a = pre_aug.augment_image(a)
            else:
                if self.val_mode == 'resize':
                    fg, a = self.resize_ratio(fg, a, use_short=False, force_resize=True)
                bg = self.resize_as(bg, fg, interpolation=cv2.INTER_CUBIC)[:fg.shape[0], :fg.shape[1]]

                if self.val_mode == 'dim':
                    # DIM re-implementation by IndexMatting will resize to 32N first
                    fg = self.resize_32(fg, interpolation=cv2.INTER_CUBIC)
                    a = self.resize_32(a, interpolation=cv2.INTER_CUBIC)
                    bg = self.resize_32(bg, interpolation=cv2.INTER_CUBIC)

                # GCA Matting uses reflect padding during evaluation
                padding_mode = 'reflect' if self.val_mode == 'gca' else 'constant'
                pad_h = self.image_shape[0] - fg.shape[0]
                pad_w = self.image_shape[1] - fg.shape[1]
                fg = np.pad(fg, ((0,pad_h), (0, pad_w), (0,0)), mode=padding_mode)
                bg = np.pad(bg, ((0,pad_h), (0, pad_w), (0,0)), mode=padding_mode)
                a = np.pad(a, ((0,pad_h), (0, pad_w)), mode=padding_mode)

            # Everything is 800x800 now
            assert fg.shape == bg.shape, a.shape[:2] == fg.shape[:2]

            fgs, bgs, alphas, good_sample = self.make_frames3(fg, bg, a)
            if not good_sample:
                del fgs
                del bgs
                del alphas
                del fg
                del bg
                del a

        #comps, tris = [None] * 3, [None] * 3
        #tri_size = np.random.randint(5, 21)
        #start = time.time()
        if self.isTrain:
            s_aug = self.shape_aug.to_deterministic()
            for i in range(3):
                fg = s_aug.augment_image(fgs[i])
                bg = s_aug.augment_image(bgs[i])
                a = s_aug.augment_image(alphas[i])
                fg = cv2.resize(fg, (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_AREA)
                bg = cv2.resize(bg, (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_CUBIC)
                a = cv2.resize(a, (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_AREA)[..., np.newaxis]
                fgs[i] = fg
                bgs[i] = bg
                alphas[i] = a
            #comps[i] = self.compose(fg, bg, a)
            #tris[i] = self.make_trimap(a, tri_size)
        #end = time.time()
        fgs = np.stack(fgs)
        bgs = np.stack(bgs)
        alphas = np.stack(alphas)
        fgt = torch.from_numpy(fgs).permute(0, 3, 1, 2).float()    # [3, 3, H, W]
        bgt = torch.from_numpy(bgs).permute(0, 3, 1, 2).float()    # [3, 3, H, W]
        at = torch.from_numpy(alphas).permute(0, 3, 1, 2).float()  # [3, 1, H, W]
        del fgs
        del bgs
        del alphas 
        #print (end-start)
        #tri = torch.from_numpy(np.stack(tris)).permute(0, 3, 1, 2).float()  # [3, 1, H, W]
        if self.isTrain:
            return at, fgt, bgt
        else:
            og_shape = torch.tensor(og_shape)
            return at, fgt, bgt, og_shape, torch.tensor(idx)
        #return rgb, tri, a, fg, bg

class DIMEvalDataset(DIMPretrainDataset):
    def __init__(self, **kwargs):
        super().__init__(isTrain=False, **kwargs)

    def make_frames3(self, fg, bg, alpha):
        def _rotate_(M, center = None, angle = None, scale = None):
            M_now = cv2.getRotationMatrix2D((center[0],center[1]), angle, scale)
            M_now = np.concatenate((M_now,[[0,0,1]]))
            return np.matmul(M, M_now)

        def _move_(M,vec):
            M_now = np.float32([[1,0,vec[0]],[0,1,vec[1]]])
            M_now = np.concatenate((M_now,[[0,0,1]]))
            return np.matmul(M,M_now)

        def _get_new_M(M, var):
            center, move, angle, scale = var
            _rM = _rotate_(M, center, angle, scale)
            _mM = _move_(_rM,move)
            return _mM
        # input: fg, bg, alpha
        # output: listfg, listbg, lista
        fgs, bgs, alphas = [None] * 3, [None] * 3, [None] * 3
        good_sample = True

        bh, bw = bg.shape[:2]
        fh, fw = fg.shape[:2]
        #dh, dw = (bh - fh) / 2., (bw - fw) / 2.
        #rh, rw = (np.random.random() - 0.5) * 2., (np.random.random() - 0.5) * 2.
        I = np.eye(3, dtype=np.float32)
        FM0 = I
        FF_var = [np.array([fw * 0.5, fh * 0.5]), np.array([0, 0]), 0, 1.0]
        FSTEP_var = [np.array([fw * 0.5, fh * 0.5]), np.array([-5, -5]), -2, 0.99]
        BSTEP_var = [np.array([fw * 0.5, fh * 0.5]), np.array([5, 5]), 2, 1.01]
        FMs = _get_new_M(I, FSTEP_var)
        BMs = _get_new_M(I, BSTEP_var)

        good_sample = True
        FM0 = np.matmul(_get_new_M(I, FF_var), FM0)
        FM_ = [np.linalg.inv(FMs), I, FMs]
        BM_ = [np.linalg.inv(BMs), I, BMs]
        for i in range(3):
            FM = np.matmul(FM_[i], FM0)
            BM = BM_[i]
            fgs[i] = cv2.warpPerspective(fg, FM, (bw, bh))
            bgs[i] = cv2.warpPerspective(bg, BM, (bw, bh))
            alphas[i] = cv2.warpPerspective(alpha, FM, (bw, bh))[..., np.newaxis]
            if np.sum(np.logical_and(alphas[i] > 0, alphas[i] < 255)) < 400:
                good_sample = False
                
        return fgs, bgs, alphas, good_sample