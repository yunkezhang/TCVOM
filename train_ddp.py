import argparse
import logging
import os
import shutil
import time
import timeit

import numpy as np
import cv2 as cv
cv.setNumThreads(0)
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as torch_dist
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision.utils import save_image
from tqdm import tqdm

from config import get_cfg_defaults
from dataset.VMD import VideoMattingDataset
from models.model import FullModel_VMD
from utils.utils import OPT_DICT, STR_DICT, \
    AverageMeter, create_logger, torch_barrier, reduce_tensor


def write_image(outdir, out, step, max_batch=4):
    with torch.no_grad():
        scaled_imgs, scaled_tris, alphas, comps, gts, fgs, bgs = out
        b, s, _, h, w = scaled_imgs.shape
        b = max_batch if b > max_batch else b
        save_image(scaled_imgs[:max_batch].reshape(b*s, 3, h, w), os.path.join(outdir, 'vis_image_{}.png'.format(step)), nrow=s)
        save_image(scaled_tris[:max_batch].reshape(b*s, 1, h, w), os.path.join(outdir, 'vis_tris_{}.png'.format(step)), nrow=s)
        save_image(alphas[:max_batch].reshape(b*s, 1, h, w), os.path.join(outdir, 'vis_as_{}.png'.format(step)), nrow=s)
        save_image(comps[:max_batch].reshape(b*s, 3, h, w), os.path.join(outdir, 'vis_comps_{}.png'.format(step)), nrow=s)
        save_image(gts[:max_batch].reshape(b*s, 1, h, w), os.path.join(outdir, 'vis_gts_{}.png'.format(step)), nrow=s)
        save_image(fgs[:max_batch].reshape(b*s, 3, h, w), os.path.join(outdir, 'vis_fgs_{}.png'.format(step)), nrow=s)
        save_image(bgs[:max_batch].reshape(b*s, 3, h, w), os.path.join(outdir, 'vis_bgs_{}.png'.format(step)), nrow=s)

def train(epoch, trainloader, steps_per_val, base_lr,
          total_epochs, optimizer, model, 
          adjust_learning_rate, print_freq, 
          image_freq, image_outdir, local_rank, sub_losses):
    # Training
    model.train()
    
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*steps_per_val
    for i_iter, dp in enumerate(trainloader):
        def handle_batch():
            fg, bg, a, _ = dp      # [B, length, 3 or 1, H, W]
            #print (a.shape)
            out = model(a, fg, bg)
            L_alpha = out[0].mean()
            L_comp = out[1].mean()
            L_grad = out[2].mean()
            L_dt = out[3].mean()
            L_att = out[4].mean()
            loss = L_alpha + L_comp + L_grad + 0.5 * L_dt + 0.25 * L_att

            model.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.detach(), L_alpha.detach(), \
                L_comp.detach(), L_grad.detach(), \
                L_dt.detach(), L_att.detach(), out[5:]

        loss, L_alpha, L_comp, L_grad, L_dt, L_att, vis_out = handle_batch()

        reduced_loss = reduce_tensor(loss)
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        torch_barrier()

        adjust_learning_rate(optimizer,
                            base_lr,
                            total_epochs * steps_per_val,
                            i_iter+cur_iters)

        if i_iter % print_freq == 0 and local_rank <= 0:
            msg = 'Iter:[{}/{}], Time: {:.2f}, '.format(\
                i_iter+cur_iters, total_epochs * steps_per_val, batch_time.average())
            msg += 'lr: {}, Avg. Loss: {:.6f} | Current: Loss: {:.6f}, '.format(
                [x['lr'] for x in optimizer.param_groups],
                ave_loss.average(), ave_loss.value())
            msg += '{}: {:.4f} {}: {:.4f} {}: {:.4f} L_dt: {:.4f} L_att: {:.4f}'.format(
                sub_losses[0], L_alpha.item(),
                sub_losses[1], L_comp.item(),
                sub_losses[2], L_grad.item(),
                L_dt.item(), L_att.item())
            logging.info(msg)
        
        if i_iter % image_freq == 0 and local_rank <= 0:
            write_image(image_outdir, vis_out, i_iter+cur_iters)
            
def validate(testloader, model, test_size, local_rank, dataset_samples, tmp_folder='/dev/shm/val_tmp'):
    if local_rank <= 0:
        logging.info('Start evaluation...')
    model.eval()
    ave_loss = AverageMeter()
    c = len(dataset_samples[0]) // 2
    
    # We calculate L_dt as a mere indicator of temporal consistency.
    # Since we have sample_length=3 during validation, which means
    # there's only one prediction (the middle frame). Thus, here we 
    # first save the prediction to tmp_folder then compute L_dt in
    # one pass.
    with torch.no_grad():
        iterator = tqdm(testloader, ascii=True) if local_rank <= 0 else testloader
        for batch in iterator:
            fg, bg, a, idx = batch      # [B, 3, 3 or 1, H, W]
            def handle_batch():
                out = model(a, fg, bg)
                L_alpha = out[0].mean()
                L_comp = out[1].mean()
                L_grad = out[2].mean()
                loss = L_alpha + L_comp + L_grad
                return loss.detach(), out[6].detach(), out[7].detach()
            
            loss, tris, alphas = handle_batch()
            reduced_loss = reduce_tensor(loss)

            for i in range(tris.shape[0]):
                fn = dataset_samples[idx[i].item()][c]
                outpath = os.path.join(tmp_folder, fn)
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                pred = np.uint8((alphas[i, c, 0] * 255).cpu().numpy())
                tri = tris[i, c, 0] * 255
                tri = np.uint8(((tri > 0) * (tri < 255)).cpu().numpy() * 255)
                gt = np.uint8(a[i, c, 0].numpy())
                out = np.stack([pred, tri, gt], axis=-1)
                cv.imwrite(outpath, out)

            ave_loss.update(reduced_loss.item())
    loss = ave_loss.average()

    if local_rank <= 0:
        logging.info('Validation loss: {:.6f}'.format(ave_loss.average()))

        def _read_output(fn):
            fn = os.path.join(tmp_folder, fn)
            preds = cv.imread(fn)
            a, m, g = np.split(preds, 3, axis=-1)
            a = np.float32(np.squeeze(a)) / 255.0
            m = np.squeeze(m) != 0
            g = np.float32(np.squeeze(g)) / 255.0
            return a, g, m 
        res = 0.
        for sample in tqdm(dataset_samples, ascii=True):
            a, g, m = _read_output(sample[c])
            ha, hg, _ = _read_output(sample[c+1])
            dadt = a - ha
            dgtdt = g - hg
            if np.sum(m) == 0:
                continue
            res += np.mean(np.abs(dadt[m] - dgtdt[m]))
        res /= float(len(dataset_samples))
        logging.info('Average L_dt: {:.6f}'.format(res))
        loss += res
        shutil.rmtree(tmp_folder)

    torch_barrier()
    return loss

def get_sampler(dataset, shuffle=True):
    if torch_dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset, shuffle=shuffle)
    else:
        return None

def main(cfg_name, cfg, local_rank):
    cfg_name += cfg.SYSTEM.EXP_SUFFIX
    random_seed = cfg.SYSTEM.RANDOM_SEED
    load_ckpt = cfg.TRAIN.LOAD_CKPT
    load_opt = cfg.TRAIN.LOAD_OPT
    base_lr = cfg.TRAIN.BASE_LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    output_dir = cfg.SYSTEM.OUTDIR
    start = timeit.default_timer()

    # cudnn related setting
    cudnn.benchmark = cfg.SYSTEM.CUDNN_BENCHMARK
    cudnn.deterministic = cfg.SYSTEM.CUDNN_DETERMINISTIC
    cudnn.enabled = cfg.SYSTEM.CUDNN_ENABLED
    if random_seed > 0:
        import random
        if local_rank <= 0:
            print('Seeding with', random_seed)
        random.seed(random_seed+local_rank)
        torch.manual_seed(random_seed+local_rank)    

    if local_rank >= 0:
        device = torch.device('cuda:{}'.format(local_rank))    
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )    
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    if local_rank <= 0:
        logger, final_output_dir = create_logger(output_dir, cfg_name, 'train')
        print (cfg)
        with open(os.path.join(final_output_dir, 'config.yaml'), 'w') as f:
            f.write(str(cfg))
        image_outdir = os.path.join(final_output_dir, 'training_images')
        os.makedirs(os.path.join(final_output_dir, 'training_images'), exist_ok=True)
    else:
        image_outdir = None

    # build model
    model = FullModel_VMD(model=cfg.MODEL, agg_window=cfg.AGG_WINDOW)
    torch_barrier()

    # prepare data
    train_dataset = VideoMattingDataset(
        data_root=cfg.DATASET.PATH,
        image_shape=cfg.TRAIN.TRAIN_INPUT_SIZE,
        mode='train',
        use_subset=cfg.DATASET.SUBSET,
        plus1=cfg.MODEL.startswith('vmn_res'),
        no_flow=True,
    )
    train_sampler = get_sampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        #shuffle=True,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    test_dataset = VideoMattingDataset(
        data_root=cfg.DATASET.PATH,
        image_shape=cfg.TRAIN.VAL_INPUT_SIZE,
        mode='val',
        use_subset=cfg.DATASET.SUBSET,
        plus1=cfg.MODEL.startswith('vmn_res'),
        no_flow=True,           # to accelerate validation during training we don't compute temporal loss
        sample_length=3,        # we also reduce the sample_length to 3
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN.VAL_BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        sampler=get_sampler(test_dataset, shuffle=False)
    )

    if load_ckpt != '':
        dct = torch.load(load_ckpt, map_location=torch.device('cpu'))
        missing_keys, unexpected_keys = model.NET.load_state_dict(dct, strict=False)
        if local_rank <= 0:
            logger.info('Missing keys: ' + str(sorted(missing_keys)))
            logger.info('Unexpected keys: ' + str(sorted(unexpected_keys)))
            logger.info("=> loaded checkpoint from {}".format(load_ckpt))
        torch_barrier()

    if local_rank >= 0:
        # FBA particularly uses batch_size == 1, thus no syncbn here
        if not cfg.MODEL.endswith('fba'):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[local_rank],
            output_device=local_rank
        )
    else:
        model = torch.nn.DataParallel(model, device_ids=[device])

    # optimizer
    params_dict = {k: v for k, v in model.named_parameters() \
        if v.requires_grad}
        
    params_count = 0
    if local_rank <= 0:
        logging.info('=> Parameters needs to be optimized:')
        for k in sorted(params_dict):
            logging.info('\t=> {}, size: {}'.format(k, list(params_dict[k].size())))
            params_count += params_dict[k].shape.numel()
        logging.info('=> Total Parameters: {}'.format(params_count))
        
    params = [{'params': list(params_dict.values()), 'lr': base_lr}]
    optimizer = OPT_DICT[cfg.TRAIN.OPTIMIZER](params, lr=base_lr, weight_decay=weight_decay)
    adjust_lr = STR_DICT[cfg.TRAIN.LR_STRATEGY]

    if load_opt != '':
        optimizer.load_state_dict(torch.load(load_opt, map_location='cpu'))
        start_step = int(os.path.basename(load_opt).split('_')[-1][:-8])
    else:
        start_step = 0

    total_steps = cfg.TRAIN.TOTAL_STEPS
    steps_per_val = len(trainloader)
    print_freq = cfg.TRAIN.PRINT_FREQ
    image_freq = cfg.TRAIN.IMAGE_FREQ
   
    #tmp_folder = '/dev/shm/val_tmp_{}_{}'.format(cfg_name, start_step)
    #validate(testloader, model, len(test_dataset), local_rank, test_dataset.samples, tmp_folder=tmp_folder)
    sub_losses = ['L_alpha', 'L_comp', 'L_grad'] if not cfg.MODEL.endswith('fba') else \
                 ['L_alpha_comp', 'L_lap', 'L_grad']
    best_loss = 1e+8
    for epoch in range(start_step, total_steps):
        if torch_dist.is_initialized():
            train_sampler.set_epoch(epoch)
        train(epoch, trainloader, steps_per_val, base_lr, total_steps,
              optimizer, model, adjust_lr, print_freq, image_freq,
              image_outdir, local_rank, sub_losses)
        torch_barrier()
        if epoch >= 15:
            tmp_folder = '/dev/shm/val_tmp_{}'.format(cfg_name, epoch)
            val_loss = validate(testloader, model, len(test_dataset), \
                local_rank, test_dataset.samples, tmp_folder=tmp_folder)
        else:
            val_loss = best_loss
        torch_barrier()

        if local_rank <= 0:
            weight_fn = os.path.join(final_output_dir,\
                'checkpoint_{}.pth.tar'.format(epoch+1))
            opt_fn = os.path.join(final_output_dir,\
                'optimizer_{}.pth.tar'.format(epoch+1))
            logger.info('=> saving checkpoint to {}'.format(weight_fn))
            logger.info('=> saving optimizer to {}'.format(opt_fn))
            torch.save(model.module.NET.state_dict(), weight_fn)
            torch.save(optimizer.state_dict(), opt_fn)
            if val_loss < best_loss:
                best_loss = val_loss
                shutil.copyfile(weight_fn, os.path.join(final_output_dir, 'best.pth'))
                logger.info('=> new minimum loss. copy to best.pth')
        
    end = timeit.default_timer()
    torch_barrier()
    if local_rank <= 0:
        logger.info('Time: %d sec.' % np.int((end-start)))
        logger.info('Done')


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return args, cfg

if __name__ == "__main__":
    args, cfg = parse_args()
    main(os.path.splitext(os.path.basename(args.cfg))[0], cfg, args.local_rank)
