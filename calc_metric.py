import argparse
import json
import multiprocessing as mp
import os
import json
from functools import partial

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

from utils.utils import coords_grid, flow_dt, grid_sampler


'''
All metric functions are in [H, W] (for single channel like a/g/m),
or [H, W, C] (for multiple channel like flow) numpy array.
m.dtype should be `bool`.
'''

def SAD(a, g, m):
    return float(np.mean(np.abs(a[m] - g[m])))

def MSE(a, g, m):
    return float(np.mean((a[m] - g[m]) ** 2))

def SSDA(a, g, m):
    return float(np.sqrt(np.sum((a[m] - g[m]) ** 2)))

def dtSSD(a, g, m, ha, hg):
    dadt = a - ha
    dgdt = g - hg
    return float(np.sqrt(np.sum((dadt[m] - dgdt[m]) ** 2)))

def MESSDdt(a, g, m, ha, hg, flow):
    # Convert all to torch tensor and calc them there.
    a = torch.from_numpy(a)[None, None, ...]
    g = torch.from_numpy(g)[None, None, ...]
    ha = torch.from_numpy(ha)[None, None, ...]
    hg = torch.from_numpy(hg)[None, None, ...]
    flow = flow.permute(2, 0, 1).unsqueeze(0)
    m = torch.from_numpy(m)[None, None, ...]

    fix, org, valid = flow_dt(a, ha, g, hg, flow, m, metric=True, cuda=False)
    return float(fix.item()), float(org.item()), int(valid.item())

def calc_metric(fn, args, print_fn=True):
    def _read_file(fn):
        ap = os.path.join(args.pred, fn+'_pred.png')
        tp = os.path.join(args.pred, fn+'_tri.png')
        gp = os.path.join(args.data, 'FG_done', fn+'.png')
        alpha = cv.imread(ap, cv.IMREAD_GRAYSCALE)
        tri = cv.imread(tp, cv.IMREAD_GRAYSCALE)
        gt = cv.imread(gp, cv.IMREAD_UNCHANGED)[..., -1]
        return alpha, tri, gt
    
    def _preprocess(alpha, gt, tri):
        a = np.float32(alpha / 255.0)
        g = np.float32(gt / 255.0)
        m = (tri > 0) * (tri < 255)
        return a, g, m

    def _flow_read(fa, fb, dn, flow_folder='flow_png'):
        x = cv.imread(os.path.join(args.data, flow_folder, dn, 'flow_{}_{}.png'.format(fa, fb)), cv.IMREAD_UNCHANGED)
        flow = np.float32(np.int16(x[..., :-1]))
        mask = x[..., -1]
        invalid = mask == 0
        flow[invalid] = np.nan
        return torch.from_numpy(flow) / 100.0
    
    if print_fn:
        print (fn[0])

    cf, nf = fn
    cfn = os.path.splitext(cf)[0]
    ca, ct, cg = _read_file(cfn)
    a, g, m = _preprocess(ca, cg, ct)
    pcount = int(np.sum(m))
    sad = SAD(a, g, m)
    mse = MSE(a, g, m)
    ssda = SSDA(a, g, m)

    if nf != '':
        nfn = os.path.splitext(nf)[0]
        ha, ht, hg = _read_file(nfn)
        ha, hg, hm = _preprocess(ha, hg, ht)

        dirbase = os.path.dirname(cfn)
        cfbase = os.path.basename(cfn)
        nfbase = os.path.basename(nfn)
        assert dirbase == os.path.dirname(nfn), '{} | {}'.format(cfn, nfn)

        flow = _flow_read(cfbase, nfbase, dirbase)
        dtssd = dtSSD(a, g, m, ha, hg)
        fixdt, orgdt, valid = MESSDdt(a, g, m, ha, hg, flow)
    else:
        fixdt = 0
        orgdt = 0
        valid = 0
        dtssd = 0

    if args.vis:
        # visulization
        vis_outdir = os.path.join(args.pred, 'vis')
        os.makedirs(os.path.join(vis_outdir, os.path.dirname(cfn)), exist_ok=True)
        vis_t = cv.resize(ct, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv.INTER_NEAREST)
        vis_t = cv.cvtColor(vis_t, cv.COLOR_GRAY2BGR)
        diff_ag = np.float32(np.abs(np.int32(ca) - np.int32(cg)))[..., np.newaxis] / 255.0
        ca = cv.cvtColor(ca, cv.COLOR_GRAY2BGR)
        cg = cv.cvtColor(cg, cv.COLOR_GRAY2BGR)
        red_mask = np.float32(np.zeros_like(ca))
        red_mask[..., -1] = 1
        red_fuse = diff_ag * red_mask + (1. - diff_ag) * np.float32(ca / 255.0)
        red_fuse = np.uint8(red_fuse * 255.0)
        comp = cv.hconcat([red_fuse, cg])
        _left = comp.shape[1] // 2 - vis_t.shape[1] // 2
        comp[-vis_t.shape[0]:, _left:_left+vis_t.shape[1]] = vis_t
        #comp = cv.cvtColor(comp, cv.COLOR_GRAY2BGR)
        comp = cv.copyMakeBorder(comp, 0, 100, 0, 0,
            cv.BORDER_CONSTANT, value=(255, 0, 0))
        s = 'SAD={:.6f} MSE={:.6f} valid_pxs={}'.format(\
                sad, mse, pcount)
        comp = cv.putText(comp, s, (20, comp.shape[0] - 20), \
            cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv.imwrite(os.path.join(vis_outdir, cfn+'.png'), comp)

    return {'mSAD':sad, 'MSE':mse, 'SSDA':ssda, 'dtSSD':dtssd, 'MESSDdt_fix':fixdt, 'MESSDdt':orgdt, \
            'pixel_count':pcount, 'flow_pixel_count':valid}

def main(args):
    # check predicted frames
    with open(os.path.join(args.data, 'frame_corr.json'), 'rb') as f:
        fdict = json.load(f)
    frame_exist = {}
    for f in sorted(fdict.keys()):
        fn = os.path.splitext(f)[0]
        frame_exist[f] = os.path.exists(os.path.join(args.pred, fn+'_pred.png')) and \
            os.path.exists(os.path.join(args.pred, fn+'_tri.png'))
    
    # check full videos
    videos = []
    current_video = ''
    full = True
    for f in sorted(fdict.keys()):
        dirn = os.path.dirname(f)
        if dirn != current_video:
            if full and current_video != '':
                videos.append(current_video)
            current_video = dirn
            full = True
        if not frame_exist[f]:
            full = False
    print ('Present videos:', videos)
    #videos = [videos[0]]                    ### FOR DEBUGGING ###

    # gather frames
    frames = []
    for f in sorted(frame_exist.keys()):
        if not frame_exist[f]:
            continue
        flag = False
        for v in videos:
            if os.path.dirname(f) == v:
                flag = True
                break
        if flag:
            frames.append(f)
    for i in range(len(frames)-1):
        cf = frames[i]
        nf = frames[i+1]
        if os.path.dirname(cf) != os.path.dirname(nf):
            frames[i] = (cf, '')
        else:
            frames[i] = (cf, nf)
    frames[-1] = (frames[-1], '')

    # calculate metric
    calc_part = partial(calc_metric, args=args)
    if args.n_threads is not None and int(args.n_threads) == 0:
        frame_result = []
        for f in tqdm(frames, ascii=True):
            frame_result.append(calc_part(f, print_fn=False))
    else:
        n_threads = args.n_threads \
            if args.n_threads is None else int(args.n_threads)
        with mp.Pool(processes=n_threads) as pool:
            frame_result = pool.map(calc_part, frames)

    # gather all metrics, calcuate per video
    results = {'avg':{}, 'all':{}}
    allres = {'mSAD':0., 'MSE':0., 'SSDA':0., 'dtSSD':0., 'MESSDdt_fix':0., 'MESSDdt':0.}
    for v in videos:
        results['all'][v] = {'avg':{}, 'all':{}}
        cres = {'mSAD':0, 'MSE':0, 'SSDA':0, 'dtSSD':0, \
                'MESSDdt_fix':0, 'MESSDdt':0, \
                'pixel_count':0, 'flow_pixel_count':0}
        
        # per video
        count = 0
        for i in range(len(frames)):
            fn = frames[i][0]
            if os.path.dirname(fn) == v:
                results['all'][v]['all'][fn] = frame_result[i]
                count += 1
                for k in frame_result[i].keys():
                    cres[k] += results['all'][v]['all'][fn][k]
        
        # video avg
        cres['mSAD'] /= float(count)
        cres['MSE'] /= float(count)
        cres['SSDA'] /= float(count)
        cres['dtSSD'] /= float(count)
        cres['MESSDdt_fix'] /= float(count)
        cres['MESSDdt'] /= float(count)
        results['all'][v]['avg'] = cres

        # add up all videos
        for k in allres.keys():
            allres[k] += cres[k]
    
    # total avg
    for k in allres.keys():
        allres[k] /= float(len(videos))
    results['avg'] = allres
    
    if args.output is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        output = args.output
    else:
        output = os.path.join(args.pred, 'metric.json')
    with open(output, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default=None, help='/path/to/metric/json/file')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--n_threads', default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser()
    main(args)    
