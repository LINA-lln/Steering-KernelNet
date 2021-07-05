import torch
from torch.autograd import Variable
if torch.cuda.is_available():
    from _ext import kernel_regression_cuda

class KernelRegression():
    def __init__(self, y_mask, h_observed, h_window, w_window, H, W, h, num_observed,
                          dtype, sigma_mat, theta_mat, scale_mat, gradbuf_sigma, gradbuf_theta,
                          gradbuf_scale, image_gt, mask, dindexes, loss_option):
        self.y_mask = y_mask
        self.h_observed = h_observed
        self.h_window = h_window
        self.w_window = w_window
        self.H = H
        self.W = W
        self.dense_depth = None
        self.h = h
        self.num_observed = num_observed
        self.dtype = dtype
        self.sigma_mat = sigma_mat
        self.theta_mat = theta_mat
        self.scale_mat = scale_mat
        self.gradbuf_sigma = gradbuf_sigma
        self.gradbuf_theta = gradbuf_theta
        self.gradbuf_scale = gradbuf_scale
        self.image_gt = image_gt.data
        self.mask = mask
        self.dindexes = dindexes.data
        self.loss_option = loss_option

    def forward_and_backward(self):
        if self.dense_depth is None:
            self.dense_depth = torch.zeros(self.H, self.W).type(self.dtype)
        else:
            self.dense_depth.zero_()  #?? whether zero?

        kernel_regression_cuda.kernel_regression_cuda_all(self.y_mask, self.h_observed,
                                                               self.h_window, self.w_window,
                                                               self.H, self.W, self.dense_depth, self.h,
                                                               self.num_observed,
                                                               self.gradbuf_sigma, self.gradbuf_theta, self.gradbuf_scale,
                                                               self.image_gt, self.sigma_mat, self.theta_mat, self.scale_mat,
                                                               self.mask, self.dindexes, self.loss_option)

        return self.dense_depth ,self.gradbuf_sigma, self.gradbuf_theta, self.gradbuf_scale



