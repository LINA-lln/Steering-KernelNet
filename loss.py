import torch
from torch.autograd import Variable 
import numpy as np
import cv2
import matplotlib as mpl
import sobel
mpl.use('Agg')
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

def compute_error(depth_gt, depth_pre, thres_set):
    ###########################
    # Average relative error  #
    npixel = np.size(depth_gt)
    rel = np.sum(np.abs(depth_gt-depth_pre)/depth_gt)/npixel
    
    ###########################
    # Root mean squared error #
    rms = np.sqrt(np.sum((depth_gt-depth_pre)**2)/npixel)
    
    ###########################
    #   Average log10 error   #
    logerr = np.sum(np.abs(np.log10(depth_gt)-np.log10(depth_pre)))/npixel
    
    ###########################
    #   Accuracy with thres   #
    if len(depth_gt.shape)==1:
        maxdivide = np.zeros([2,depth_gt.shape[0]])
    elif len(depth_gt.shape)==2:
        maxdivide = np.zeros([2,depth_gt.shape[0],depth_gt.shape[1]])

    maxdivide[0] = depth_gt/depth_pre
    maxdivide[1] = depth_pre/depth_gt
    maxdivide = np.max(maxdivide,axis=0)
    thres_ind = 0
    awt = np.zeros(np.shape(thres_set))
    for thres in thres_set:
        within_thres = np.where(maxdivide<thres)
        awt[thres_ind] = float(len(within_thres[0]))/float(npixel)
        thres_ind = thres_ind + 1
    
    return (rel, rms, logerr, awt)

class Loss():
    def __init__(self, args):

        self.loss_type = args.loss
        self.variance = args.variance
        self.num_bin = args.num_bin 
        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        
        self.weight_sign = args.weight_sign
        self.weight_regression = args.weight_regression
        self.weight_variance = args.weight_variance
        self.weight_class = args.weight_class
        self.dataset = args.dataset
        self.observation = args.observation
        self.grad_loss = args.grad_loss
        self.get_gradient = sobel.Sobel().cuda()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=0)
        print ("Init Loss Object")

    # calculate the expectation depth value given the discretized estimation
    # input
    #   est .... NxCxHxW
    # output
    #   mean ... NxHxW 
    def depth_expectation(self, est):

        N = est.size()[0]
        C = est.size()[1]
        H = est.size()[2]
        W = est.size()[3]

        est = est.transpose(0,1)
        est = est.contiguous().view(C, N*H*W)

        # apply softmax to normalize the probability
        softmax = torch.nn.Softmax()
        est = softmax(est.transpose(0,1)).transpose(0,1)

        # depth values corresponding to each bin
        # divided by (self.num_bin-1) as the first bin is 0
        values = Variable(torch.arange(0, self.num_bin).unsqueeze(1))
        values = values / (self.num_bin-1) * (self.max_depth - self.min_depth) + self.min_depth

        if torch.cuda.is_available():
            values = values.cuda()
        exp = torch.sum(torch.mul(est, values), dim=0)
        exp = exp.view(N, H, W)
        return exp , est
    
    # calculate the variance given the discretized estimation and the mean 
    # input
    #   est .... NxCxHxW
    #   mean ... NxHxW 
    # output
    #   var .... NxHxW
    def depth_variance(self, est, mean, mask=None):
        N = est.size()[0]
        C = est.size()[1]
        H = est.size()[2]
        W = est.size()[3]

        est = est.transpose(0,1).contiguous().view(C, N*H*W)
        mean = mean.view(1, N*H*W)

        # apply softmax to normalize the probability
        softmax = torch.nn.Softmax()
        est = softmax(est.transpose(0,1)).transpose(0,1)

        # depth values corresponding to each bin
        # divided by (self.num_bin-1) as the first bin is 0
        values = Variable(torch.arange(0, self.num_bin).unsqueeze(1))
        values = values / (self.num_bin-1) * (self.max_depth - self.min_depth) + self.min_depth
        #values = values.expand(C, N*H*W)
        values = values.view(C, 1)

        if torch.cuda.is_available():
            values = values.cuda()

        # variances at points with valid observations
        if mask is not None:
            mask_ = torch.nonzero(mask.type(torch.ByteTensor).data.view(-1))
            mask_ = mask_.cuda().view(-1)
            mean_masked = mean[:, mask_]
            est_masked = est[:, mask_]
            values_masked = values - mean_masked
            values_masked = torch.mul(values_masked, values_masked)
            var_masked = torch.sum(torch.mul(est_masked, values_masked), dim=0)
            return var_masked
	
        # variances of the full images
        values = values - mean
        values = torch.mul(values, values)
        var = torch.sum(torch.mul(est, values), dim=0)

        var = var.view(N, H, W)

        return var

    def loss_on_classification(self, est, gt, ignored_label=255):
        N = est.size()[0]
        C = est.size()[1]
        H = est.size()[2]
        W = est.size()[3]

        # NxCxHxW to Cx(NxHxW) to (NxHxW)xC
        est = est.transpose(0,1).contiguous().view(C, N*H*W)
        est = est.transpose(0,1).contiguous()
        loss = torch.nn.CrossEntropyLoss(ignore_index=ignored_label)
        return loss(est, gt.view(-1))
    
    def loss_on_regression(self, est, gt, mask):
        loss = torch.mean(torch.abs(est-gt))  #l1
        return loss

    def loss_on_sobel_regression(self, est, gt, mask, edges):
        loss = torch.sum(torch.mul(torch.mul(torch.abs(est - gt), mask), edges)) / torch.sum(mask)
        return loss

    def loss_on_variance(self, est, mean, mask):
        var_masked = self.depth_variance(est, mean, mask)
        #loss = torch.mean(torch.mul(var,mask))
        loss = torch.mean(var_masked)
        return loss

    def train_loss(self):
        pass

    def eval_metrics(self, est, est_sign, gt_float, args, sparse_rnd,residual=False, reference=None, percents=[100.]):
        est_float, est_softmax = self.depth_expectation(est)
        est_var = self.depth_variance(est, est_float)
        est_float = est_float.data.cpu().numpy()
        gt_float = gt_float.data.cpu().numpy()
        est_var = est_var.data.cpu().numpy()
        est_sign = est_sign.data.cpu().numpy()

        est_float = np.squeeze(est_float)
        gt_float = np.squeeze(gt_float)
        est_var = np.squeeze(est_var)
        
        # if in residual mode add reference to the residual
        if residual > 0:

            reference = np.squeeze(reference.data.cpu().numpy()).astype('float64')

            # if self.dataset == 'nyud2':
            ref_mean = 3.01047486221
            ref_std = 1.07943839977

            reference = reference* ref_std + ref_mean

            if residual == 1:
                est_float = est_float + reference

                if args.sign_loss:
                    est_float = est_float + est_sign.squeeze()  # * 0.01   ##add
                    est_float = Variable(torch.from_numpy(est_float)).type(dtype)
                    est_float[sparse_rnd > 0] = sparse_rnd[sparse_rnd>0]
                    est_float = est_float.data.cpu().numpy().squeeze()

            # clamp the depth value to a reasonable range for save log operation
            if self.dataset == 'nyud2':
               est_float[est_float<0.7] = 0.7
               est_float[est_float>10.0] = 10.0

        est_float = cv2.resize(est_float, (gt_float.shape[1], gt_float.shape[0]), interpolation=cv2.INTER_NEAREST)
        est_var = cv2.resize(est_var, (gt_float.shape[1], gt_float.shape[0]), interpolation=cv2.INTER_NEAREST)
        assert(est_float.shape==gt_float.shape)

        est_return = est_float
        gt_mask = gt_float > 0
        metrics = []

        est_float_full = est_float[gt_mask] 
        gt_float_full = gt_float[gt_mask]
        results_full = compute_error(gt_float_full, est_float_full, [1.02, 1.05, 1.10, 1.25, 1.25**2, 1.25**3] )
        metrics.append(results_full)

        rel = np.asarray([m[0] for m in metrics])
        rms = np.asarray([m[1] for m in metrics])
        log = np.asarray([m[2] for m in metrics])
        awt = np.asarray([m[3] for m in metrics])

        return (rms, rel, log, awt), est_return, est_var, est_softmax

