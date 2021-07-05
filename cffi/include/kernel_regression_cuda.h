int kernel_regression_cuda_all( THCudaTensor *y_mask_tensor, THCudaTensor *h_observed_tensor, int h_window, int w_window,
                              int H, int W, THCudaTensor *dense_depth_tensor, float h,
                              int num_observed, THCudaTensor *gradbuf_sigma_tensor, THCudaTensor *gradbuf_theta_tensor,
                              THCudaTensor *gradbuf_scale_tensor, THCudaTensor *image_gt_tensor,
                              THCudaTensor *sigma_mat_tensor, THCudaTensor *theta_mat_tensor, THCudaTensor *scale_mat_tensor,
                               THCudaTensor *mask_tensor,  THCudaTensor *dindexes_tensor, int loss_option);


