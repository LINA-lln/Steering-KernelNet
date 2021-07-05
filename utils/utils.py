import numpy as np

def depth_to_log(depth):
    sign = depth<0
    #depth_log = np.log2(1+np.abs(depth))
    depth_log = np.log(1+np.abs(depth))
    depth_log[sign] = -depth_log[sign]
    return depth_log

def log_to_depth(depth_log):
    sign = depth_log<0
    #depth = np.exp2(np.abs(depth_log)) - 1
    depth = np.exp(np.abs(depth_log)) - 1
    depth[sign] = -depth[sign]
    return depth

# discretize the depth into a given number of bins
def depth_discretization(depth, nbin, dmin=0, dmax=100):
    # linear discretization
    # multiplied by (nbin-1) as the first bin is 0
    return np.round((depth - dmin)/(dmax - dmin) * (nbin-1)).astype(np.int32)
