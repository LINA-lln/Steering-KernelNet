#include <THC.h>
#include <THCGeneral.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "cuComplex.h"
#include <curand.h>
#include "math.h"
#include <cuda.h>
#include "cublas_v2.h"
//#include "cusolverDn.h"
#include "../include/kernel_regression_kernel.h"
//#include <device_functions.h>

#include <time.h>
#include <sys/time.h>
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)
#define INDEX 200 //130
#ifdef __cplusplus
extern "C" {
#endif

__device__ static double PYTHAG(double a, double b)
{
    double at = fabs(a), bt = fabs(b), ct, result;

    if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
    else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
    else result = 0.0;
    return(result);
}


__device__ void dsvd(float a[], int m, int n, float w[], float v[], float inv_XTWxX[])
{
    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;
    double rv1[6];

    if (m < n)
    {
        printf("error: #rows must be > #cols \n");
        return;
    }

    // rv1 = (double *)malloc((unsigned int)n*sizeof(double));

    // Householder reduction to bidiagonal form
    for (i = 0; i < n; i++)
    {
        // left-hand reduction
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < m)
        {
            for (k = i; k < m; k++)
                scale += fabs((double)a[k*n+i]);
            if (scale)
            {
                for (k = i; k < m; k++)
                {
                    a[k*n+i] = (float)((double)a[k*n+i] / scale);
                    s += ((double)a[k*n+i] * (double)a[k*n+i]);
                }
                f = (double)a[i*n+i];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[i*n+i] = (float)(f - g);
                if (i != n - 1)
                {
                    for (j = l; j < n; j++)
                    {
                        for (s = 0.0, k = i; k < m; k++)
                            s += ((double)a[k*n+i] * (double)a[k*n+j]);
                        f = s / h;
                        for (k = i; k < m; k++)
                            a[k*n+j] += (float)(f * (double)a[k*n+i]);
                    }
                }
                for (k = i; k < m; k++)
                    a[k*n+i] = (float)((double)a[k*n+i] * scale);
            }
        }
        __syncthreads();
        w[i] = (float)(scale * g);

        //right-hand reduction
        g = s = scale = 0.0;
        if (i < m && i != n - 1)
        {
            for (k = l; k < n; k++)
                scale += fabs((double)a[i*n+k]);
            if (scale)
            {
                for (k = l; k < n; k++)
                {
                    a[i*n+k] = (float)((double)a[i*n+k] / scale);
                    s += ((double)a[i*n+k] * (double)a[i*n+k]);
                }
                f = (double)a[i*n+l];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[i*n+l] = (float)(f - g);
                for (k = l; k < n; k++)
                    rv1[k] = (double)a[i*n+k] / h;
                if (i != m - 1)
                {
                    for (j = l; j < m; j++)
                    {
                        for (s = 0.0, k = l; k < n; k++)
                            s += ((double)a[j*n+k] * (double)a[i*n+k]);
                        for (k = l; k < n; k++)
                            a[j*n+k] += (float)(s * rv1[k]);
                    }
                }
                for (k = l; k < n; k++)
                    a[i*n+k] = (float)((double)a[i*n+k] * scale);
            }
        }
        anorm = MAX(anorm, (fabs((double)w[i]) + fabs(rv1[i])));
    }

    // accumulate the right-hand transformation
    for (i = n - 1; i >= 0; i--)
    {
        if (i < n - 1)
        {
            if (g)
            {
                for (j = l; j < n; j++)
                    v[j*n+i] = (float)(((double)a[i*n+j] / (double)a[i*n+l]) / g);
                //double division to avoid underflow
                for (j = l; j < n; j++)
                {
                    for (s = 0.0, k = l; k < n; k++)
                        s += ((double)a[i*n+k] * (double)v[k*n+j]);
                    for (k = l; k < n; k++)
                        v[k*n+j] += (float)(s * (double)v[k*n+i]);
                }
            }
            for (j = l; j < n; j++)
                v[i*n+j] = v[j*n+i] = 0.0;
        }
        v[i*n+i] = 1.0;
        g = rv1[i];
        l = i;
    }

    // accumulate the left-hand transformation
    for (i = n - 1; i >= 0; i--)
    {
        l = i + 1;
        g = (double)w[i];
        if (i < n - 1)
            for (j = l; j < n; j++)
                a[i*n+j] = 0.0;
        if (g)
        {
            g = 1.0 / g;
            if (i != n - 1)
            {
                for (j = l; j < n; j++)
                {
                    for (s = 0.0, k = l; k < m; k++)
                        s += ((double)a[k*n+i] * (double)a[k*n+j]);
                    f = (s / (double)a[i*n+i]) * g;
                    for (k = i; k < m; k++)
                        a[k*n+j] += (float)(f * (double)a[k*n+i]);
                }
            }
            for (j = i; j < m; j++)
                a[j*n+i] = (float)((double)a[j*n+i] * g);
        }
        else
        {
            for (j = i; j < m; j++)
                a[j*n+i] = 0.0;
        }
        ++a[i*n+i];
    }

    // diagonalize the bidiagonal form
    for (k = n - 1; k >= 0; k--)
    {                             // loop over singular values
        for (its = 0; its < 30; its++)
        {                         // loop over allowed iterations
            flag = 1;
            for (l = k; l >= 0; l--)
            {                     // test for splitting
                nm = l - 1;
                if (fabs(rv1[l]) + anorm == anorm)
                {
                    flag = 0;
                    break;
                }
                if (fabs((double)w[nm]) + anorm == anorm)
                    break;
            }
            if (flag)
            {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++)
                {
                    f = s * rv1[i];
                    if (fabs(f) + anorm != anorm)
                    {
                        g = (double)w[i];
                       // if(fabs(f)>fabs(g)){h = fabs(f)*sqrt(1.0+fabs(g)*fabs(g)/(fabs(f)*fabs(f)));}
                       // else if(fabs(g)>0.0) {h = fabs(g)*sqrt(1.0+fabs(f)*fabs(f)/(fabs(g)*fabs(g)));}
                       // else h = 0.0;
                        h = PYTHAG(f, g);
                        w[i] = (float)h;
                        h = 1.0 / h;
                        c = g * h;
                        s = (-f * h);
                        for (j = 0; j < m; j++)
                        {
                            y = (double)a[j*n+nm];
                            z = (double)a[j*n+i];
                            a[j*n+nm] = (float)(y * c + z * s);
                            a[j*n+i] = (float)(z * c - y * s);
                        }
                    }
                }
            }
            z = (double)w[k];
            if (l == k)
            {                  // convergence
                if (z < 0.0)
                {              // make singular value nonnegative
                    w[k] = (float)(-z);
                    for (j = 0; j < n; j++)
                        v[j*n+k] = (-v[j*n+k]);
                }
                break;
            }
            if (its >= 30) {
               // cudaFree(rv1);
                printf("error: No convergence after 30,000! iterations \n");
                return;
            }

            // shift from bottom 2 x 2 minor
            x = (double)w[l];
            nm = k - 1;
            y = (double)w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = PYTHAG(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

            // next QR transformation
            c = s = 1.0;
            for (j = l; j <= nm; j++)
            {
                i = j + 1;
                g = rv1[i];
                y = (double)w[i];
                h = s * g;
                g = c * g;
    
                z = PYTHAG(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++)
                {
                    x = (double)v[jj*n+j];
                    z = (double)v[jj*n+i];
                    v[jj*n+j] = (float)(x * c + z * s);
                    v[jj*n+i] = (float)(z * c - x * s);
                }
              
                z = PYTHAG(f, h);
                w[j] = (float)z;
                if (z)
                {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++)
                {
                    y = (double)a[jj*n+j];
                    z = (double)a[jj*n+i];
                    a[jj*n+j] = (float)(y * c + z * s);
                    a[jj*n+i] = (float)(z * c - y * s);
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = (float)x;
        }
    }
   // cudaFree(rv1);

   for(i=0; i<6; i++){
            for(j=0; j<6;j++) {
                if (abs(w[j]) > 1e-5)
                    v[i * 6 + j] = v[i * 6 + j] / w[j];
                else
                    v[i * 6 + j] = 0;
            }
        }
        for(i=0;i<6;i++){
            for(j=0;j<6;j++){
                inv_XTWxX[i*6+j] = v[i*6+0]*a[j*6+0]+v[i*6+1]*a[j*6+1]+v[i*6+2]*a[j*6+2]+v[i*6+3]*a[j*6+3]+v[i*6+4]*a[j*6+4]+v[i*6+5]*a[j*6+5];
            }
        }
}


__global__ void kr_kernel(const float *d_y_mask_data, float *d_dense_depth_data,
                          const float *d_h_observed_data, const int *d_H, const int *d_W, float *d_indexes,
                          const int *dh_window, const int *dw_window, const float *d_h,
                          const float *d_image_gt_data,
                          const float *d_sigma_data, const float *d_theta_data, const float *d_scale_data, float *d_grad_sigma,
                          float *d_grad_theta, float *d_grad_scale, const float *d_mask_data, const int *d_loss_option)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // printf("%d ", blockDim.x);
    int nx = *d_W;
    int ny = *d_H;
    int n_loss_option = *d_loss_option;
    if(idx < nx*ny){

        int nh_window = *dh_window;
        int nw_window = *dw_window;
        float dh = *d_h;
        float uh = -1/(2*dh*dh);
     // if(idx == 500)
	//  printf("%d, %d,%d, %d, %.1f\n",nx, ny, nh_window, nw_window, dh);
        //steering
        float d_sigma_curr[INDEX];
        float d_theta_curr[INDEX];
        float d_scale_curr[INDEX];
        float d_C_curr1[INDEX];
	float d_C_curr2[INDEX];
	float d_C_curr4[INDEX];
        float d_detC_curr[INDEX]; //sqrt

        int q =0;
        for(int ii = idx/nx; ii < (idx/nx+2*nh_window+1); ii++){
            for(int jj = idx%nx; jj < (idx%nx+2*nw_window+1); jj++){
                if(d_mask_data[ii*(nx+2*nw_window)+jj] != 0 ){  //this is error, change y_mask to mask  //&& q<(int)d_indexes[idx]

                    d_sigma_curr[q] = d_sigma_data[ii*(nx+2*nw_window)+jj];
                    
                            if(d_sigma_curr[q] >= 0){
                                d_sigma_curr[q] +=1e-5;
                            }
                            else
                                d_sigma_curr[q] -=1e-5;
                    
                    d_theta_curr[q] = d_theta_data[ii*(nx+2*nw_window)+jj];
                    d_scale_curr[q] = d_scale_data[ii*(nx+2*nw_window)+jj];
                    q++;
                }
            }
        }
        
        d_indexes[idx] = q;

        for(int is = 0; is < (int)d_indexes[idx]; is++){
            d_C_curr1[is] = d_scale_curr[is] * d_sigma_curr[is] * pow(cos(d_theta_curr[is]) ,2) + d_scale_curr[is] / d_sigma_curr[is] * pow(sin(d_theta_curr[is]) ,2);
            d_C_curr2[is] = d_scale_curr[is] / d_sigma_curr[is] * cos(d_theta_curr[is]) * sin(d_theta_curr[is]) - d_scale_curr[is] * cos(d_theta_curr[is]) * sin(d_theta_curr[is]);
          //  d_C_curr[4*is+2] = d_C_curr[4*is+1];
            d_C_curr4[is] = d_scale_curr[is] * d_sigma_curr[is] * pow(sin(d_theta_curr[is]) ,2) + d_scale_curr[is] / d_sigma_curr[is] * pow(cos(d_theta_curr[is]) ,2);
            d_detC_curr[is] = abs(d_scale_curr[is]);
	   // printf("%.1f %.1f %.1f--- ",d_C_curr[4*is],d_C_curr[4*is+1],d_C_curr[4*is+3]);

        }

//kr
        float d_Kh[INDEX];
        float d_x_diff1[INDEX];
	float d_x_diff2[INDEX];
        float d_y_mask_curr[INDEX];
        float d_h_observed_curr[INDEX];


        float d_xtcx[INDEX];
	//float d_tanh2[INDEX];
        __syncthreads();



        int kk =0;
        int flag_x = 0;
        int flag_y = 0;
        for(int i = idx/nx; i < (idx/nx+2*nh_window+1); i++){
            for(int j = idx%nx; j < (idx%nx+2*nw_window+1); j++){
                if(d_mask_data[i*(nx+2*nw_window)+j] != 0 ){  //this is error, change y_mask to mask //&& k<(int)d_indexes[idx]

                    d_y_mask_curr[kk] = d_y_mask_data[i*(nx+2*nw_window)+j];

                    d_x_diff1[kk] = i - (idx/nx + nh_window);
                    d_x_diff2[kk] = j - (idx%nx + nw_window);
                    if(d_x_diff1[kk]==0 && d_x_diff2[kk]==0){
                        flag_x = i;
                        flag_y = j;
                    }

                    d_xtcx[kk] = d_x_diff1[kk] * d_x_diff1[kk] * d_C_curr1[kk] \
                               + 2*d_x_diff1[kk] * d_x_diff2[kk] * d_C_curr2[kk] + \
                               d_x_diff2[kk] * d_x_diff2[kk] * d_C_curr4[kk];
		    
                    d_Kh[kk] = d_detC_curr[kk] * exp(2*tanh(uh * d_xtcx[kk]));   //exp error: too large, because C is too large
		  


                    d_h_observed_curr[kk] = d_h_observed_data[i*(nx+2*nw_window)+j];

                    kk++;
                }
            }
        }

        __syncthreads();
        float XTWxX[36];

        float temp1[15];
        float temp2[6];


        for(int it = 0; it<15; it++){
            temp1[it] = 0.0;
            if(it<6){
                temp2[it] = 0.0;
            }
        }
        for(int ia = 0; ia<d_indexes[idx]; ia++){
            temp1[0] +=  d_Kh[ia];
            temp1[1] += (d_Kh[ia] * d_x_diff1[ia]);
            temp1[2] += (d_Kh[ia] * d_x_diff2[ia]);
            temp1[3] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff1[ia]);
            temp1[4] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff2[ia]);
            temp1[5] += (d_Kh[ia] * d_x_diff2[ia] * d_x_diff2[ia]);
            temp1[6] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff1[ia] * d_x_diff1[ia]);
            temp1[7] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff1[ia] * d_x_diff2[ia]);
            temp1[8] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff2[ia] * d_x_diff2[ia]);
            temp1[9] += (d_Kh[ia] * d_x_diff2[ia] * d_x_diff2[ia] * d_x_diff2[ia]);
            temp1[10] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff1[ia] * d_x_diff1[ia] * d_x_diff1[ia]);
            temp1[11] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff1[ia] * d_x_diff1[ia] * d_x_diff2[ia]);
            temp1[12] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff1[ia] * d_x_diff2[ia] * d_x_diff2[ia]);
            temp1[13] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff2[ia] * d_x_diff2[ia] * d_x_diff2[ia]);
            temp1[14] += (d_Kh[ia] * d_x_diff2[ia] * d_x_diff2[ia] * d_x_diff2[ia] * d_x_diff2[ia]);
	    
	   // printf("%.1f ",d_Kh[ia]); //d_Kh has error: a large data, then inf...

            temp2[0] += (d_Kh[ia] * d_y_mask_curr[ia]);
            temp2[1] += (d_Kh[ia] * d_x_diff1[ia] * d_y_mask_curr[ia]);
            temp2[2] += (d_Kh[ia] * d_x_diff2[ia] * d_y_mask_curr[ia]);
            temp2[3] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff1[ia] * d_y_mask_curr[ia]);
            temp2[4] += (d_Kh[ia] * d_x_diff1[ia] * d_x_diff2[ia] * d_y_mask_curr[ia]);
            temp2[5] += (d_Kh[ia] * d_x_diff2[ia] * d_x_diff2[ia] * d_y_mask_curr[ia]);

        }
        __syncthreads();
        XTWxX[0] = temp1[0];// +1e-5;
        XTWxX[1] = XTWxX[6] = temp1[1];
        XTWxX[2] = XTWxX[12] = temp1[2];
        XTWxX[3] = XTWxX[7] = XTWxX[18] = temp1[3];
        XTWxX[4] = XTWxX[8] = XTWxX[13] = XTWxX[24] = temp1[4];
        XTWxX[5] = XTWxX[14] = XTWxX[30] = temp1[5];
        XTWxX[9] = XTWxX[19] = temp1[6];
        XTWxX[10] = XTWxX[15] = XTWxX[20] = XTWxX[25] =temp1[7];
        XTWxX[11] = XTWxX[16] = XTWxX[26] = XTWxX[31] = temp1[8];
        XTWxX[17] = XTWxX[32] =temp1[9];
        XTWxX[21] =temp1[10];
        XTWxX[22] = XTWxX[27] = temp1[11];
        XTWxX[23] = XTWxX[28] = XTWxX[33] = temp1[12];
        XTWxX[29] = XTWxX[34] = temp1[13];
        XTWxX[35] = temp1[14];
	
        //inverse
        __syncthreads();
        float inv_XTWxX[36];
        float a[36];
        float v[36];
        float w[6];

        __syncthreads();

	for(int i=0;i<36;i++){
	    a[i] = XTWxX[i];
	    inv_XTWxX[i] = 0.0;
	}
	

//===========================================================
        dsvd(a,6,6,w,v, inv_XTWxX);


//===========================================================
        __syncthreads();
        
        d_dense_depth_data[idx] = 0.0;
        if(flag_x==0 || flag_y==0){
            for(int id = 0; id<6; id++){
                d_dense_depth_data[idx] += (inv_XTWxX[id] * temp2[id]);
            }
        }

        if(flag_x!=0 && flag_y!=0){
            d_dense_depth_data[idx] = d_y_mask_data[(idx/nx +nh_window)*(nx+2*nw_window)+(idx%nx+nw_window)];
        }


        //backward
        //grad loss/zx
        float grad_loss_zx;
        if(n_loss_option == 1){
        if(d_image_gt_data[idx] <= d_dense_depth_data[idx])
            grad_loss_zx = 1;///(nx*ny);
        else
            grad_loss_zx = -1;///(nx*ny);
        }
        else if(n_loss_option == 2){  //loss = 1/2*(y-gt)^2
            grad_loss_zx = d_dense_depth_data[idx] - d_image_gt_data[idx];

        }

        //grad zx/khi
        __syncthreads();
        float grad_zx_kh[INDEX];
        float temp3[6];

        //add
        float temp4[6];
	
        if(flag_x==0 || flag_y==0){
            for(int i = 0; i < 6; i++){
	        temp4[i] = 0.0;  //must be init
                for(int j = 0; j < 6; j++){
                    temp4[i] += (inv_XTWxX[i*6+j]*temp2[j]);
                }
            }
        }

        __syncthreads();
        float temp5[6];
        float temp6[36];
        float grad_zx_kh2[INDEX];
    
        if(flag_x==0 || flag_y==0){
            for(int ik = 0; ik< (int)d_indexes[idx]; ik++){
                grad_zx_kh[ik] = 0.0;
                grad_zx_kh2[ik] = 0.0;
                temp3[0] = 1.0;
                temp3[1] = d_x_diff1[ik];
                temp3[2] = d_x_diff2[ik];
                temp3[3] = d_x_diff1[ik] * d_x_diff1[ik] ;
                temp3[4] = d_x_diff1[ik] * d_x_diff2[ik] ;
                temp3[5] = d_x_diff2[ik] * d_x_diff2[ik];
                for(int jk = 0; jk<6;jk++){
                    grad_zx_kh[ik] += (inv_XTWxX[jk] * temp3[jk]);
                }
                // grad_zx_kh[ik] *= (d_y_mask_curr[ik] - d_dense_depth_data[idx]);
                // change new
                //first
                grad_zx_kh[ik] *= d_y_mask_curr[ik];
                //second

                temp6[0] = 1.0;
                temp6[1] = temp6[6] = d_x_diff1[ik];
                temp6[2] = temp6[12] = d_x_diff2[ik];
                temp6[3] = temp6[7] = temp6[18] = d_x_diff1[ik] * d_x_diff1[ik] ;
                temp6[4] = temp6[8] = temp6[13] = temp6[24] = d_x_diff1[ik] * d_x_diff2[ik] ;
                temp6[5] = temp6[14] = temp6[30] = d_x_diff2[ik] * d_x_diff2[ik];
                temp6[9] = temp6[19] = d_x_diff1[ik]*d_x_diff1[ik] * d_x_diff1[ik];
                temp6[10] = temp6[15] = temp6[20] = temp6[25] =d_x_diff1[ik]*d_x_diff1[ik] * d_x_diff2[ik];
                temp6[11] = temp6[16] = temp6[26] = temp6[31] = d_x_diff1[ik]*d_x_diff2[ik] * d_x_diff2[ik];
                temp6[17] = temp6[32] = d_x_diff2[ik]*d_x_diff2[ik] * d_x_diff2[ik];
                temp6[21] = d_x_diff1[ik] * d_x_diff1[ik] * d_x_diff1[ik] * d_x_diff1[ik];
                temp6[22] = temp6[27] = d_x_diff1[ik] * d_x_diff1[ik] * d_x_diff1[ik] * d_x_diff2[ik];
                temp6[23] = temp6[28] = temp6[33] = d_x_diff1[ik] * d_x_diff1[ik] * d_x_diff2[ik] * d_x_diff2[ik];
                temp6[29] = temp6[34] = d_x_diff1[ik] * d_x_diff2[ik] * d_x_diff2[ik] * d_x_diff2[ik];
                temp6[35] = d_x_diff2[ik] * d_x_diff2[ik] * d_x_diff2[ik] * d_x_diff2[ik];//+1e-5;
                for(int p = 0; p<6; p++){
		            temp5[p] = 0.0;
                    for(int q=0; q<6; q++){
                        temp5[p] += (temp6[p*6+q]*temp4[q]); //1/(1+exp(-temp6[p*6+q]))
                    }
                }
                for(int jp = 0; jp<6;jp++){
                    grad_zx_kh2[ik] += (inv_XTWxX[jp] * temp5[jp]);
                }
                grad_zx_kh[ik] = grad_zx_kh[ik] - grad_zx_kh2[ik];

            }
        }
	
        //grad khi/theta sigma and scale
        float grad_kh_scale[INDEX];
        float grad_kh_theta[INDEX];
        float grad_kh_sigma[INDEX];
	    //float grad_xtcx[INDEX];
        __syncthreads();
        if(flag_x==0 || flag_y==0){
            for(int ig = 0; ig<d_indexes[idx]; ig++){

                grad_kh_scale[ig] = d_scale_curr[ig]/(d_detC_curr[ig]*d_detC_curr[ig]+1e-5)*d_Kh[ig] + d_Kh[ig]*  2*(1-tanh(uh*d_xtcx[ig])*tanh(uh*d_xtcx[ig]))  *(-0.5/(dh*dh))*d_xtcx[ig] / (d_scale_curr[ig]+1e-5);
                grad_kh_sigma[ig] = d_Kh[ig]*  2*(1-tanh(uh*d_xtcx[ig])*tanh(uh*d_xtcx[ig]))  *(-0.5/(dh*dh)) * d_scale_curr[ig] * (pow((d_x_diff1[ig]*cos(d_theta_curr[ig])-d_x_diff2[ig]*sin(d_theta_curr[ig])), 2) \
                            - 1/(d_sigma_curr[ig]*d_sigma_curr[ig]) *(pow((d_x_diff1[ig]*sin(d_theta_curr[ig])+d_x_diff2[ig]*cos(d_theta_curr[ig])), 2)) );

                grad_kh_theta[ig] = d_Kh[ig]*  2*(1-tanh(uh*d_xtcx[ig])*tanh(uh*d_xtcx[ig]))  *(-0.5/(dh*dh)) *d_scale_curr[ig] * (1/d_sigma_curr[ig] - d_sigma_curr[ig]) * \
                            ( d_x_diff1[ig]*d_x_diff1[ig]*sin(2*d_theta_curr[ig]) + 2*d_x_diff1[ig]*d_x_diff2[ig]*cos(2*d_theta_curr[ig]) - d_x_diff2[ig]*d_x_diff2[ig]*sin(2*d_theta_curr[ig]));

            }
        }

        //change grad to each pixel, now is simple sum
        
        __syncthreads();

        if(flag_x==0 || flag_y==0){
            for(int iz = 0; iz<d_indexes[idx]; iz++){
                d_grad_sigma[(int)d_h_observed_curr[iz]] += (grad_loss_zx* grad_zx_kh[iz] * grad_kh_sigma[iz] );//* grad_zx_kh[iz] * grad_kh_sigma[iz]);
                d_grad_theta[(int)d_h_observed_curr[iz]] += (grad_loss_zx* grad_zx_kh[iz] * grad_kh_theta[iz] );//* grad_zx_kh[iz] * grad_kh_theta[iz]);
                d_grad_scale[(int)d_h_observed_curr[iz]] += (grad_loss_zx* grad_zx_kh[iz] * grad_kh_scale[iz]);//* grad_zx_kh[iz] * grad_kh_scale[iz]);
            }
        }

        __syncthreads();
	
	//cudaFree(U);
	//cudaFree(V);
	//cudaFree(w);      

    }
}


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void create_indexes(float *mask_data, int nh_window, int nw_window, int *indexes, int nH, int nW){

    // *sum_indexes = 0;
    for(int i = nh_window; i < nH + nh_window; i++){
        for(int j = nw_window; j < nW + nw_window; j++){
            indexes[(i-nh_window) * nW +(j-nw_window)] = 0;
            for(int p = i-nh_window; p < i+nh_window+1; p++){
                for(int q = j-nw_window; q < j+nw_window+1; q++){
                    if(mask_data[p*(nW+2*nw_window) + q] != 0){
                        indexes[(i-nh_window) * nW +(j-nw_window)] ++ ;
                    }}}
        }}
}


void Kernel_regression_kernel(float *y_mask_data, float *h_observed_data,
                              int *h_window, int *w_window, int *H, int *W, float *dense_depth_data,
                              float *h, int *num_observed, float *grad_sigma,
                              float *grad_theta, float *grad_scale, float *image_gt_data,
                              float *sigma_data, float *theta_data,  float *scale_data, float *mask_data,
                              float *indexes_data,int *loss_option, cudaStream_t stream){

    int nH = *H;
    int nW = *W;
    //int nh_window = *h_window;
    //int nw_window = *w_window;
    //int nxy = (nH+nh_window*2) * (nW+nw_window*2);
    //int num_ob = *num_observed;

    int *d_H, *d_W, *dh_window, *dw_window, *d_loss_option;
    float *d_h;
    cudaMalloc((void **) &d_H, sizeof(int));
    cudaMalloc((void **) &d_W, sizeof(int));
    cudaMalloc((void **) &dh_window, sizeof(int));
    cudaMalloc((void **) &dw_window, sizeof(int));
    cudaMalloc((void **) &d_loss_option, sizeof(int));
    cudaMalloc((void **) &d_h, sizeof(float));

    cudaMemcpy(d_H, H, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dh_window, h_window, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dw_window, w_window, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss_option, loss_option, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h, sizeof(float), cudaMemcpyHostToDevice);

    // cublasHandle_t handle;
    // cublasCreate(&handle);
    dim3 dimGrid(nH*2);
    dim3 dimBlock(nW/2);  //block should not be too large, will get too many resources .....

    double iStart = cpuSecond();

    kr_kernel <<< dimGrid, dimBlock, 0, stream>>>(y_mask_data, dense_depth_data,
            h_observed_data, d_H, d_W, indexes_data, dh_window, dw_window, d_h,
            image_gt_data,sigma_data, theta_data, scale_data,
            grad_sigma, grad_theta, grad_scale, mask_data, d_loss_option);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    // if(cudaGetErrorString(error)!= "no error")
    // printf("CUDA1 error: %s\n", cudaGetErrorString(error));
    double iElaps = cpuSecond() - iStart;
    //   printf("Kernel Use Time:%.3f s\n",iElaps);//(stop-start)

    cudaFree(d_H);
    cudaFree(d_W);
    cudaFree(dh_window);
    cudaFree(dw_window);
    cudaFree(d_h);
    // cudaDeviceReset();  //invalid

    cudaStreamSynchronize(stream);

}

#ifdef __cplusplus
}
#endif


 

