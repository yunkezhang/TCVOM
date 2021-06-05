import argparse
import multiprocessing as mp
import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import EvalModel

class TestFolder(object):
    SAMPLE_LENGTH = 3
    def __init__(self, data_root, videos):
        self.data_root = data_root
        if videos == []:
            videos = [f for f in sorted(glob.glob(os.path.join(args.data, '*')))\
                if os.path.isdir(f)]
        print (videos)

        # parse dir
        def _make_samples():
            vdict = {}
            for v in videos:
                src = sorted(glob.glob(os.path.join(data_root, v, '*_rgb.png')))
                tri = sorted(glob.glob(os.path.join(data_root, v, '*_trimap.png')))
                vdict[v] = list(zip(src, tri))  # src=0, tri=1

            # convert to samples
            samples = []
            for v in sorted(vdict.keys()):
                for c in range(len(vdict[v])):
                    p = c+1 if c == 0 else c-1
                    n = c-1 if c == len(vdict[v])-1 else c+1
                    samples.append((vdict[v][p], vdict[v][c], vdict[v][n]))
            return samples

        self.samples = _make_samples()

    def __len__(self):
        return len(self.samples)

    def possible_pad(self, t, padvalue=None):
        H, W = t.shape[-2:]
        NH, NW = int(np.ceil(H / 32.0) * 32), int(np.ceil(W / 32.0) * 32)
        t = t.float()
        if H == NH and W == NW:
            return t
        ph, pw = NH - H, NW - W
        if isinstance(padvalue, (int, float)):
            return F.pad(t.unsqueeze(0), (0, pw, 0, ph), value=padvalue).squeeze(0)
        elif isinstance(padvalue, (list, tuple)):
            assert len(padvalue) == t.shape[-3]
            mask = F.pad(torch.zeros(H, W), (0, pw, 0, ph), value=1).bool()
            t = F.pad(t, (0, pw, 0, ph), value=padvalue)
            v = torch.tensor(padvalue, dtype=t.dtype).unsqueeze(-1)
            t[:, mask] = v
            return t
        elif padvalue is None:
            return F.pad(t.unsqueeze(0), (0, pw, 0, ph), mode='reflect').squeeze(0)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = []
        tris = []
        for i in range(self.SAMPLE_LENGTH):
            imgs.append(cv2.imread(sample[i][0], cv2.IMREAD_UNCHANGED))
            tris.append(cv2.imread(sample[i][1], cv2.IMREAD_GRAYSCALE)[..., np.newaxis])
        og_shape = cv2.imread(sample[0][0]).shape[:2]

        for i in range(self.SAMPLE_LENGTH):
            imgs[i] = self.possible_pad(torch.from_numpy(imgs[i]).permute(2, 0, 1))
            tris[i] = self.possible_pad(torch.from_numpy(tris[i]).permute(2, 0, 1))

        imgs = torch.stack(imgs).float()
        tris = torch.stack(tris).float()
        return imgs, tris, torch.tensor(og_shape)
        
def pred(dataset, indices, device, args):
    torch.cuda.set_device(device)
    c = dataset.SAMPLE_LENGTH // 2
    start, end = indices
    model = EvalModel(model=args.model, \
        agg_window=args.agg_window, dilate_kernel=args.dilation)
    model.NET.load_state_dict(torch.load(args.load, map_location='cpu'), strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for _id in range(start, end):
            def handle_batch():
                imgs, tris, og_shape = dataset[_id]
                H, W = og_shape.numpy()
                imgs = imgs.to(device).unsqueeze(0)
                tris = tris.to(device).unsqueeze(0)
                info = os.path.normpath(dataset.samples[_id][c][0])
                info = info.split(os.sep)
                if args.model.endswith('fba'):
                    preds, Fs, Bs = model(imgs, tris)
                    pred = preds.squeeze()[c][:H, :W].detach().cpu().numpy()
                else:
                    pred = model(imgs, tris).squeeze()[c][:H, :W].detach().cpu().numpy()
                outfn = os.path.join(args.save, info[-2], info[-1][:-8]+'_alpha.png')
                return pred, outfn

            pred, outfn = handle_batch()
            print (outfn, device.index, _id, end)
            os.makedirs(os.path.dirname(outfn), exist_ok=True)
            cv2.imwrite(outfn, np.uint8(pred * 255))

def main(args):
    if args.save is None:
        args.save = 'test_results/{}'.format(os.path.splitext(args.load)[0])
    os.makedirs(args.save, exist_ok=True)
    dataset = TestFolder(args.data, args.videos)
    gpus = args.gpu.split(',')
    if len(gpus) != 1:
        pproc = len(dataset) // len(gpus) + 1
        ps = []
        for i, gid in enumerate(gpus):
            ps.append(mp.Process(target=pred, args=(dataset,
                (i*pproc, min((i+1)*pproc, len(dataset))),
                torch.device('cuda:{}'.format(gid)),
                args)),
            )

        for p in ps:
            p.start()
        for p in ps:
            p.join()
    else:
        pred(dataset, (0, len(dataset)), torch.device('cuda:{}'.format(gpus[0])), args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', help='/path/to/outdir')
    parser.add_argument('--model', required=True, help='model name')
    parser.add_argument('--load', required=True, help='ckpt')
    parser.add_argument('--data', required=True, help="input data location")
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--agg_window', default=7, type=int)
    parser.add_argument('--dilation', default=None, type=int)
    parser.add_argument('videos', nargs='*')
    args = parser.parse_args()

    main(args)
