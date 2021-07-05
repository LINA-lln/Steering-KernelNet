#include <THC/THC.h>
#include <THCGeneral.h>
#include <stdio.h>
#include "cublas_v2.h"
#include "../include/kernel_regression_kernel.h"
extern THCState* state;
/*
 * Forward function, project the point features to cells, perform max pooling in every cell 
 * params: 
 *        y 		input, sparse input, HxW
 *  	  I		input, the mask indicating if there is a valid value on y, HxW 
 *  	  smooth 	input, global smoothing parameter
 *  	  covmat 	input, the covariance matrices containing local orientation information 
 *  	  k1		input, width of the kernel 
 *  	  k2 		input, height of the kernel
 *  	  zd     	output, dense output, HxW 
 *
 * Adapted from the Matlab code by Hiro
 */
int kernel_regression_cuda_all( THCudaTensor *y_mask_tensor, THCudaTensor *h_observed_tensor, int h_window, int w_window,
                              int H, int W, THCudaTensor *dense_depth_tensor, float h,
                              int num_observed, THCudaTensor *gradbuf_sigma_tensor, THCudaTensor *gradbuf_theta_tensor,
                              THCudaTensor *gradbuf_scale_tensor, THCudaTensor *image_gt_tensor,
                              THCudaTensor *sigma_mat_tensor, THCudaTensor *theta_mat_tensor, THCudaTensor *scale_mat_tensor,
                               THCudaTensor *mask_tensor,  THCudaTensor *dindexes_tensor, int loss_option)
{
  int *my_h_window = &h_window;
  int *my_w_window = &w_window;
  int *my_H = &H;
  int *my_W = &W;
  int *my_num_observed = &num_observed;
  int *my_loss_option = &loss_option;
  float *my_h = &h;

  float *y_mask_data = THCudaTensor_data(state, y_mask_tensor);
  float *h_observed_data = THCudaTensor_data(state,h_observed_tensor);  //must be float
  float *dense_depth_data = THCudaTensor_data(state,dense_depth_tensor);

  float *grad_sigma = THCudaTensor_data(state,gradbuf_sigma_tensor);
  float *grad_theta = THCudaTensor_data(state,gradbuf_theta_tensor);
  float *grad_scale = THCudaTensor_data(state,gradbuf_scale_tensor);
  float *image_gt_data = THCudaTensor_data(state,image_gt_tensor);
  float *sigma_data = THCudaTensor_data(state,sigma_mat_tensor);
  float *theta_data = THCudaTensor_data(state,theta_mat_tensor);
  float *scale_data = THCudaTensor_data(state,scale_mat_tensor);

  float *mask_data = THCudaTensor_data(state,mask_tensor);
  float *indexes_data = THCudaTensor_data(state, dindexes_tensor);

  cudaStream_t stream = THCState_getCurrentStream(state);

 // printf("kr_kernel in C started!\n");
  Kernel_regression_kernel(y_mask_data, h_observed_data, my_h_window, my_w_window, my_H, my_W,
                                    dense_depth_data, my_h, my_num_observed, grad_sigma, grad_theta, grad_scale,
                                    image_gt_data, sigma_data, theta_data, scale_data, mask_data, indexes_data,my_loss_option, stream);
 // printf("kr_kernel in C finished!\n");

  return 1;


}

