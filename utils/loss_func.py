import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def L1_mask(x, y, mask=None, epsilon=1.001e-5, normalize=True):
    res = torch.abs(x - y)
    b,c,h,w = y.shape
    if mask is not None:
        res = res * mask
        if normalize:
            _safe = torch.sum((mask > epsilon).float()).clamp(epsilon, b*c*h*w+1)
            return torch.sum(res) / _safe
        else:
            return torch.sum(res)
    if normalize:
        return torch.mean(res)
    else:
        return torch.sum(res)


def L1_mask_hard_mining(x, y, mask):
    input_size = x.size()
    res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
    with torch.no_grad():
        idx = mask > 0.5
        res_sort = [torch.sort(res[i, idx[i, ...]])[0] for i in range(idx.shape[0])]
        res_sort = [i[int(i.shape[0] * 0.5)].item() for i in res_sort]
        new_mask = mask.clone()
        for i in range(res.shape[0]):
            new_mask[i, ...] = ((mask[i, ...] > 0.5) & (res[i, ...] > res_sort[i])).float()

    res = res * new_mask
    final_res = torch.sum(res) / torch.sum(new_mask)
    return final_res, new_mask
    
def get_gradient(image):
    b, c, h, w = image.shape
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    
    dy = F.pad(dy, (0, 0, 0, 1))
    dx = F.pad(dx, (0, 1, 0, 0))
    return dx, dy

def L1_grad(pred, gt, mask=None, epsilon=1.001e-5, normalize=True):
    fake_grad_x, fake_grad_y = get_gradient(pred)
    true_grad_x, true_grad_y = get_gradient(gt)

    mag_fake = torch.sqrt(fake_grad_x ** 2 + fake_grad_y ** 2 + epsilon)
    mag_true = torch.sqrt(true_grad_x ** 2 + true_grad_y ** 2 + epsilon)

    return L1_mask(mag_fake, mag_true, mask=mask, normalize=normalize)

'''
Ported from https://github.com/ceciliavision/perceptual-reflection-removal/blob/master/main.py
'''
def exclusion_loss(img1, img2, level, epsilon=1.001e-5, normalize=True):
    gradx_loss=[]
    grady_loss=[]
    for l in range(level):
        gradx1, grady1 = get_gradient(img1)
        gradx2, grady2 = get_gradient(img2)
        
        alphax=2.0*torch.mean(torch.abs(gradx1))/(torch.mean(torch.abs(gradx2)) + epsilon)
        alphay=2.0*torch.mean(torch.abs(grady1))/(torch.mean(torch.abs(grady2)) + epsilon)
            
        gradx1_s=(torch.sigmoid(gradx1)*2)-1
        grady1_s=(torch.sigmoid(grady1)*2)-1
        gradx2_s=(torch.sigmoid(gradx2*alphax)*2)-1
        grady2_s=(torch.sigmoid(grady2*alphay)*2)-1

        safe_x = torch.mean((gradx1_s ** 2) * (gradx2_s ** 2), dim=(1,2,3)) + epsilon
        safe_y = torch.mean((grady1_s ** 2) * (grady2_s ** 2), dim=(1,2,3)) + epsilon
        gradx_loss.append(safe_x ** 0.25)
        grady_loss.append(safe_y ** 0.25)

        img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
        img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

    if normalize:
        return torch.mean(sum(gradx_loss) / float(level)) + torch.mean(sum(grady_loss) / float(level))
    else:
        return torch.sum(sum(gradx_loss) / float(level)) + torch.sum(sum(grady_loss) / float(level))

def sparsity_loss(prediction, trimask, eps=1e-5, gamma=0.9):
    mask = trimask > 0.5
    pred = prediction[mask]
    loss = torch.sum(torch.pow(pred+eps, gamma) + torch.pow(1.-pred+eps, gamma) - 1.)
    return loss

'''
Borrowed from https://gist.github.com/alper111/b9c6d80e2dba1ee0bfac15eb7dad09c8
It directly follows OpenCV's image pyramid implementation pyrDown() and pyrUp().
Reference: https://docs.opencv.org/4.4.0/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff
'''
class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                    [4., 16., 24., 16., 4.],
                    [6., 24., 36., 24., 6.],
                    [4., 16., 24., 16., 4.],
                    [1., 4., 6., 4., 1.]])
        kernel /= 256.
        self.register_buffer('KERNEL', kernel.float())

    def downsample(self, x):
        # rejecting even rows and columns
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        # Padding zeros interleaved in x (similar to unpooling where indices are always at top-left corner)
        # Original code only works when x.shape[2] == x.shape[3] because it uses the wrong indice order
        # after the first permute
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
        cc = cc.permute(0,1,3,2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
        x_up = cc.permute(0,1,3,2)
        return self.conv_gauss(x_up, 4*self.KERNEL.repeat(x.shape[1], 1, 1, 1))

    def conv_gauss(self, img, kernel):
        img = F.pad(img, (2, 2, 2, 2), mode='reflect')
        out = F.conv2d(img, kernel, groups=img.shape[1])
        return out

    def laplacian_pyramid(self, img):
        current = img
        pyr = []
        for level in range(self.max_levels):
            filtered = self.conv_gauss(current, \
                self.KERNEL.repeat(img.shape[1], 1, 1, 1))
            down = self.downsample(filtered)
            up = self.upsample(down)
            diff = current-up
            pyr.append(diff)
            current = down
        return pyr

    def forward(self, img, tgt, mask=None, normalize=True):
        pyr_input  = self.laplacian_pyramid(img)
        pyr_target = self.laplacian_pyramid(tgt)
        loss = sum((2 ** level) * L1_mask(ab[0], ab[1], mask=mask, normalize=False) \
                    for level, ab in enumerate(zip(pyr_input, pyr_target)))
        if normalize:
            b,c,h,w = tgt.shape
            if mask is not None:
                _safe = torch.sum((mask > 1e-6).float()).clamp(epsilon, b*c*h*w+1)
            else:
                _safe = b*c*h*w
            return loss / _safe
        return loss
