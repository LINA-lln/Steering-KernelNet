import torch
import warnings
warnings.filterwarnings('ignore')
from torch.autograd import Variable 
import numpy as np
import os
import matplotlib.pyplot as plt
from config import parse_args
from RefCreate import ref_create
from dataloaders import get_loader
import time
from networks import UNet
from networks import KernelNet
from loss import Loss

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor

def depth_expectation(est):
    N = est.size()[0]
    C = est.size()[1]
    H = est.size()[2]
    W = est.size()[3]

    est = est.transpose(0, 1)
    est = est.contiguous().view(C, N * H * W)

    # apply softmax to normalize the probability
    softmax = torch.nn.Softmax()
    est = softmax(est.transpose(0, 1)).transpose(0, 1)

    # depth values corresponding to each bin
    # divided by (self.num_bin-1) as the first bin is 0
    values = Variable(torch.arange(0, 401).unsqueeze(1))
    values = values / (401 - 1) * (4 + 4) - 4

    if torch.cuda.is_available():
        values = values.cuda()
    exp = torch.sum(torch.mul(est, values), dim=0)
    exp = exp.view(N, H, W)
    return exp, est

def run_val(testData, model, loss_obj, epoch, args, net):

    # evaluation
    num_test = len(testData)
    print("Testing with %d" % num_test)

    percents = [100.0]
    num_percents = 1 # 1+4 w.r.t. distance
    REL = np.zeros((num_test, num_percents)) 
    RMS = np.zeros((num_test, num_percents)) 
    LOG = np.zeros((num_test, num_percents)) 
    AWT = np.zeros((num_test, num_percents, 6))

    debug_dir = os.path.join(args.output_dir, 'debug')
    if not os.path.isdir(debug_dir):
        os.makedirs(debug_dir)

    bins = np.arange(-90, 91, 180.0/1000.0)
    hist = np.zeros(len(bins)-1).astype(np.int)

    all1 = 0
    num1 = 0  
    time1_all = time2_all = 0
    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    print('# model parameters:', sum(param.numel() for param in model.parameters()))
    for itest, sample_batched in enumerate(testData):

        if itest==0 and torch.cuda.is_available():
            model.cuda()
            model.eval()

        image_rnd = Variable(sample_batched['image'].type(dtype), volatile=True)
        gt_float_rnd = Variable(sample_batched['label'].type(dtype), volatile=True)
        reference_rnd = torch.zeros(image_rnd.size(0), args.height, args.width).type(dtype)
        sparse_rnd = torch.zeros(image_rnd.size(0), args.height, args.width).type(dtype)
        gt_mask_rnd = torch.zeros(image_rnd.size(0), args.height, args.width).type(torch.ByteTensor)
        gt_sign = torch.zeros(image_rnd.size(0), args.height, args.width)
        start1 = time.time()
        for number in range(image_rnd.size(0)):
            image_rnd[number, :, :, :], gt_float_rnd[number, :, :], reference_rnd[number, :, :], sparse_rnd[number, :,:], \
                                gt_mask_rnd[number,:, :], gt_sign[number, :,:], time1 = ref_create(image_rnd[number, :, :, :],
                                                        gt_float_rnd[number, :, :], args.height, args.width, net, args, 'val')
        reference_rnd = Variable(reference_rnd.type(dtype))
        sparse_rnd = Variable(sparse_rnd).type(dtype)
        start2 = time.time()

        if args.observation=='none':
            net_input = image_rnd
        else:
            net_input = torch.cat((image_rnd, reference_rnd.unsqueeze(1)), dim=1)

        depth_est, depth_sign = model(net_input)

        end = time.time()
        time2 = end-start2
        # print(time1, time2)
        time1_all = (time1_all * num1 + time1)/(num1+1)
        time2_all = (time2_all * num1 + time2)/(num1+1)
        all1 += end
        num1 +=1

        metrics, depth_est_float, depth_est_var, est_softmax = loss_obj.eval_metrics(depth_est, depth_sign, gt_float_rnd, args, sparse_rnd,args.residual, reference_rnd, percents)
        RMS[itest] = metrics[0] 
        REL[itest] = metrics[1] 
        LOG[itest] = metrics[2] 
        AWT[itest] = metrics[3]

        # output
        if itest % 10 == 0:
            print('evaluate %d/%d, RMS: %f, REL: %f, LOG: %f, delta1.02: %f' % (itest, num_test, RMS[itest], REL[itest], LOG[itest], AWT[itest][0][0]))

        gt_float_numpy = np.squeeze(gt_float_rnd.data.cpu().numpy())
        gt_mask = gt_float_numpy>0
        error_map = gt_float_numpy - depth_est_float
        error_map_valid = error_map[gt_mask]
        error_map[(1-gt_mask).astype(np.bool)] = 0
        hist_i, bin_edge = np.histogram(error_map_valid, bins) 
        hist += hist_i

        # vislization
        # if np.mod(itest, 1000)==0:
        #     ref_mean = 3.01047486221
        #     ref_std = 1.07943839977
        #     reference_rnd = reference_rnd* ref_std + ref_mean
        #     reference_numpy = reference_rnd.data.cpu().numpy()
        #     vmin = np.min(gt_float_numpy)
        #     vmax = np.max(gt_float_numpy)
        #     image = image_rnd.data.squeeze().cpu().numpy().transpose(1,2,0)
        #     img_mean = [0.485, 0.456, 0.406]
        #     img_std = [0.229, 0.224, 0.225]
        #     image = (image*img_std + img_mean) * 255.0
        #
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_img.png' % itest), image.astype(np.uint8))
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_gt.png' % itest ), np.squeeze(gt_float_numpy), cmap='jet', vmin=vmin, vmax=vmax)
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_spa.png' % itest), np.squeeze(sparse_rnd.data.cpu().numpy()), cmap='jet', vmin=vmin, vmax=vmax)
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_est.png' % itest ), np.squeeze(depth_est_float), cmap='jet', vmin=vmin, vmax=vmax)
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_res.png' % itest), np.squeeze(np.abs(depth_est_float-reference_numpy)), cmap='jet', vmin=np.min(np.squeeze(np.abs(depth_est_float-reference_numpy))), vmax=np.max(np.squeeze(np.abs(depth_est_float-reference_numpy))))
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_ref.png' % itest), np.squeeze(reference_numpy), cmap='jet',vmin=vmin, vmax=vmax)
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_err.png' % itest ), np.squeeze(error_map), cmap='jet', vmin=vmin, vmax=vmax)
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_var_log.png' % itest ), np.squeeze(np.log10(depth_est_var+1)), cmap='jet')
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_var_org.png' % itest ), np.squeeze(depth_est_var), cmap='jet')
        #     plt.imsave(os.path.join(debug_dir, 'val_%04d_var.png' % itest ), np.squeeze(depth_est_var), cmap='jet', vmin=np.min(np.squeeze(depth_est_var)), vmax=np.max(np.squeeze(depth_est_var)))


    print( '')
    print ('========= Evaluation results ========')
    print ('RMS:    ',np.mean(RMS, axis=0))
    print ('REL:    ',np.mean(REL, axis=0))
    print ('LogErr: ',np.mean(LOG, axis=0))
    print ('AWT:    ',np.mean(AWT, axis=0))

    np.savez(os.path.join(args.output_dir,'snapshot_%d_eval.npz') % epoch, REL, RMS, LOG, AWT, hist)
    # print('time: ',all1*1.0/num1)
    print('time1:', time1, 'time2:', time2)

    return (np.mean(RMS, axis=0)[0], np.mean(REL, axis=0)[0], np.mean(LOG, axis=0)[0], np.mean(AWT, axis=0)[0,:])


if __name__ == '__main__':

    # parse args
    args = parse_args()
    assert (os.path.isfile(args.model))

    net = KernelNet(n_channels=3, n_classes=3)
    net.load_state_dict(torch.load(args.loadref))
    #  print('Ref Model loaded from {}'.format(load))
    net.cuda()
    net.eval()
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        #  print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    eval_list = 'datalist/nyudepth_hdf5_val.csv'
    val_loader = get_loader(args.dataset, eval_list, args.data_dir, 1, args.residual, args, 'val')

    ###
    num_val = len(val_loader)
    print("Loaded %d validation samples" % num_val)

    # initialize model and loss
    ResOfResNet = UNet(n_channels=args.input_channel, n_classes=args.num_bin)
    # ResOfResNet.load_state_dict(torch.load(args.model))
    if args.model and os.path.isfile(args.model):
        print("Initialized model from %s" % args.model)
        pretrained_dict = torch.load(args.model)
        model_dict = ResOfResNet.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        ResOfResNet.load_state_dict(model_dict)

    L = Loss(args)

    ResOfResNet.load_state_dict(torch.load(args.model))
    metrics = run_val(val_loader, ResOfResNet, L, 0, args, net)


