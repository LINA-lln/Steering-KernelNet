# -*- coding: utf-8 -*
import argparse
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--phase',
                        help='Phase of this call, e.g., train/val.',
                        default='val', choices=['train','val'], type=str)
    parser.add_argument('--dataset',
                        help='Training dataset',
                        default='nyud2', choices=['nyud2','kitti'], type=str)
    parser.add_argument('--model', 
                        help='Snapshotted model, for resuming training or validation',
                        required=True, type=str)
    parser.add_argument('--loss', 
                        help='Loss function for training, e.g. classification, regression, joint',
                        default='joint', choices=['classification', 'regression', 'joint'], type=str)  #joint

    parser.add_argument('--observation', 
                        help='Type of sparse observation, e.g. singlescan, multiscan, random',
                        default='random', choices=['none', 'singlescan', 'multiscan4', 'multiscan8', 'random', 'singlescan_orig'], type=str)
    parser.add_argument('--residual', 
                        help='2: learn the propotion, 1: learn the residual, 0: learn the direct depth',
                        default=1, type=int)
    parser.add_argument('--variance', 
                        help='If add constrain on variance or not',
                        default=False, type=str2bool)
    parser.add_argument('--num_bin', 
                        help='The number of discretized bins',
                        default=401, type=int)
    parser.add_argument('--augmentation', 
                        help='If train with data augmentation or not',
                        default=True, type=str2bool)
    parser.add_argument('--ref_sparse', 
                        help='If use sparse observation as reference or not',
                        default=False, type=str2bool)
    parser.add_argument('--resnet_type', 
                        help='Loss function for training, e.g. classification, regression, joint',
                        default='resnet50', choices=['resnet50', 'resnet18'], type=str)

    #
    parser.add_argument('--data_dir', 
                        help='Input directory', 
                        required=True, type=str)
    parser.add_argument('--output_dir', 
                        help='Output directory', 
                        default='output', type=str)

    #
    parser.add_argument('--learning_rate', 
                        help='Learning rate for network training',
                        default=1e-4, type=float)
    parser.add_argument('--weight_decay', 
                        help='Weight decay for network training',
                        default=1e-3, type=float)

    #
    parser.add_argument('--height',
                        help='height of images on NYUv2',
                        default=228, type=int)
    parser.add_argument('--width',
                        help='width of images on NYUv2',
                        default=304, type=int)

    #
    parser.add_argument('--epoch', 
                        help='Number of epochs',
                        default=40, type=int)
    parser.add_argument('--snapshot', 
                        help='Snapshotting intervals',
                        default=-6, type=int)
    parser.add_argument('--batchsize', 
                        help='Number of samples in a single batch',
                        default=1, type=int)
    parser.add_argument('--multi_gpu',
                        help='If use sobel filter in loss or not',
                        default=False, type=str2bool)
    parser.add_argument('--grad_loss',
                        help='If use grad anf normal in loss or not',
                        default=True, type=str2bool)
    #
    parser.add_argument('--weight_variance', 
                        help='Weight on the variance loss',
                        default=0, type=float)
    parser.add_argument('--weight_class', 
                        help='Weight on the classification loss',
                        default=1.0, type=float)  #1
    parser.add_argument('--weight_regression', 
                        help='Weight on the regresssion loss',
                        default=1.0, type=float)  #None
    parser.add_argument('--weight_sign', 
                        help='Weight on the sign',
                        default=0.0, type=float)  #1
    parser.add_argument('--sign_loss',
                        help='If use sobel filter in loss or not',
                        default=True, type=str2bool)

    # 
    parser.add_argument('--debug', 
                        help='Debug mode, load a subset of the dataset',
                        default=False, type=str2bool)

    # ref
    parser.add_argument('-c', '--loadref', dest='loadref',
                      required=True, help='load file model')
    parser.add_argument('--num_ob', dest='numob', type=int,
                      default=500, help='number of the samples')
    parser.add_argument('--smooth', dest='smooth', type=float,
                      default=5.0, help='h of the guassin kernel')
    parser.add_argument('--h_window', dest='h_window', type=int,
                      default=26, help='h_window of kernel regression')
    parser.add_argument('--w_window', dest='w_window', type=int,
                      default=34, help='w_window of kernel regression')
    parser.add_argument('--loss_option', help='Type of loss',
                      default=1, type=int)
    parser.add_argument('--record', help='Type of loss',
                        default=0, type=int)



    args = parser.parse_args()
    print(args)

    #
    if args.observation != 'none':
        args.input_channel = 4
    else:
        args.input_channel = 3

    # default weight for different dataset
    if (args.dataset == 'kitti') and args.weight_regression is None:
        args.weight_regression = 0.1
    elif args.dataset == 'nyud2' and args.weight_regression is None:
        args.weight_regression = 1.0

    # default snapshot and epoch for different dataset
    if args.dataset == 'kitti':
        if args.snapshot < 0:
            args.snapshot = 10 
        if args.epoch < 0:
            args.epoch = 151
    elif args.dataset == 'nyud2':
        if args.snapshot < 0:
            args.snapshot = 1
        if args.epoch < 0:
            args.epoch = 31


    # specify min_depth and max_depth for discretization
    if args.dataset == 'nyud2':
        if args.residual: #is True:
            args.max_depth = 4
            args.min_depth = -4

        else:
            args.max_depth = 10
            args.min_depth = 0.7

    # regression loss doesn't provide mean and variance prediction
    assert( not (args.loss=='regression' and args.variance==True) ) 

    # check if the input directory is valid and make the output_dir if not exists
    assert(os.path.isdir(args.data_dir)) 
    #_nopretrained   unet_nopretrained_3
    args.output_dir = '%s/%s/%s_%s_sparse%d_variance%d_residual%d_lr%.06f_wd%.06f_regress%.06f_grad%s_eval' % (args.output_dir, args.dataset, args.observation, args.loss, args.ref_sparse, args.variance, args.residual, args.learning_rate, args.weight_decay, args.weight_regression, args.grad_loss)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    return args

