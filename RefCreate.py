import torch
import numpy as np
from torch.autograd import Variable
import sys
sys.path.append('..')
from functions.kr_cuda import KernelRegression
import random
import torch._utils
import cv2
from utils import log_to_depth, depth_to_log
import time

def edge_mirror(x, h, w):
    if isinstance(x, torch.FloatTensor) or isinstance(x, torch.cuda.FloatTensor) or isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor):
        xh = x.size()[0]
        xw = x.size()[1]
        if x.is_cuda:
            dtype_long = torch.cuda.LongTensor
        else:
            dtype_long = torch.LongTensor
        y = torch.cat( (x[:, torch.arange(w,0,-1).type(dtype_long)], x, x[:, torch.arange(xw-2,xw-w-2,-1).type(dtype_long)]), dim=1 )
        return torch.cat( (y[torch.arange(h,0,-1).type(dtype_long), :], y, y[torch.arange(xh-2,xh-h-2,-1).type(dtype_long), :]), dim=0 )
    elif isinstance(x, np.ndarray):
        y = np.concatenate( (x[:, w:0:-1], x, x[:, -2:-w-2:-1]), axis=1 )
        return np.concatenate( (y[h:0:-1, :], y, y[-2:-h-2:-1, :]), axis=0 )
    else:
        raise("Not Implemented Data Type for Edge Mirroring!")

def kernel_regression_val(parms_pred, y_mask, h_observed, w_observed, true_depth, mask, args, dtype, height, weight):

    parms_pred = parms_pred.view(3,height, weight)
    sigma = parms_pred[0,:,:] * mask
    theta = parms_pred[1,:,:] * mask
    scale = parms_pred[2,:,:] * mask
    indexes = Variable(torch.zeros(height, weight).type(dtype))
    h_observed_mat = Variable(torch.zeros(height, weight).type(dtype))
    for ii in range(args.numob):
        h_observed_mat[h_observed[ii], w_observed[ii]] = ii

    y_mask = edge_mirror(y_mask.data, args.h_window, args.w_window)
    mask = edge_mirror(mask.data, args.h_window, args.w_window)
    sigma_mat = edge_mirror(sigma.data, args.h_window, args.w_window)
    theta_mat = edge_mirror(theta.data, args.h_window, args.w_window)
    scale_mat = edge_mirror(scale.data, args.h_window, args.w_window)
    h_observed_mat = edge_mirror(h_observed_mat.data, args.h_window, args.w_window)
    grad_sigma = torch.zeros(args.numob).type(dtype)
    grad_theta = torch.zeros(args.numob).type(dtype)
    grad_scale = torch.zeros(args.numob).type(dtype)
    kr = KernelRegression(y_mask, h_observed_mat, args.h_window, args.w_window, height, weight, args.smooth,
                          args.numob, dtype, sigma_mat, theta_mat, scale_mat, grad_sigma, grad_theta,
                          grad_scale, true_depth, mask, indexes, args.loss_option)
    ref_depth, grad_sigma, grad_theta, grad_scale = kr.forward_and_backward()
    return ref_depth

def ref_create(img,true_depth, height, weight, net, args, phase):
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    ref_mean = 3.01047486221
    ref_std = 1.07943839977
    dtype = torch.cuda.FloatTensor
    img = img.cuda()
    true_depth = true_depth.cuda()
    img = img.expand(1, -1, -1, -1)

    ind_observed = np.array(random.sample(range(0, height * weight), args.numob))
    h_observed = ind_observed // weight
    w_observed = ind_observed % weight

    y_mask = Variable(torch.zeros(height, weight).type(dtype))
    y_mask[h_observed, w_observed] = true_depth[h_observed, w_observed]
    mask = Variable(torch.zeros(height, weight).type(dtype))
    mask[h_observed, w_observed] = 1.0

    start1 = time.time()
    parms_pred = net(img)
    ref_depth = kernel_regression_val(parms_pred[0, :, :, :], y_mask, h_observed,
                                      w_observed, true_depth, mask, args, dtype, height, weight)

    end = time.time()-start1
    # flip
    image = img.data.cpu().numpy()
    image = np.squeeze(image)
    image = np.transpose(image, (1,2,0))
    image = image.astype(float) * 255.0
    label = true_depth.data.cpu().numpy()
    ref = ref_depth.cpu().numpy()
    y_mask = y_mask.data.cpu().numpy()

    rnd_flip = random.randint(0, 1)
    if rnd_flip == 1 and phase=='train':
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
        ref = cv2.flip(ref, 1)
        y_mask = cv2.flip(y_mask, 1)
    # downsample
    mask = (label > 0).astype(np.uint8)
    # initialize the sign of the label
    label_sign = np.zeros(label.shape)
    if args.residual == 1 and phase=='train':   #do not use 2 and do not use log2
        ref[ref < 0] = 0
        ref[ref > 80] = 80
        label = label - ref

        if args.log2:
            label = depth_to_log(label)

    #normalize
    image = (image.astype(np.float32) / 255.0 - img_mean) / img_std
    ref = (ref - ref_mean) / ref_std

    return Variable(torch.from_numpy(image.transpose(2,0,1))).type(dtype), Variable(torch.from_numpy(label)).type(dtype), \
           torch.from_numpy(ref), torch.from_numpy(y_mask), torch.from_numpy(mask), torch.from_numpy(label_sign), end
