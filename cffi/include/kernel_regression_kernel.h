#ifdef __cplusplus
extern "C" {
#endif

void Kernel_regression_kernel(  float *y_mask_data,
                                     float *h_observed_data,
                                     int *h_window, int *w_window, int *H, int *W, float *dense_depth_data,
                                     float *h, int *num_observed, float *grad_sigma,
                                     float *grad_theta, float *grad_scale, float *image_gt_data,
                                     float *sigma_data, float *theta_data,  float *scale_data, float *mask_data, float *indexes_data,int *loss_option, cudaStream_t stream);


#ifdef __cplusplus
}
#endif
 

