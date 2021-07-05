#!/bin/bash

if [ ! -d build  ];then
  mkdir build
else
  echo build dir exist
fi

set -e 

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"
export C_INCLUDE_PATH=/usr/local/cuda/lib64:$C_INCLUDE_PATH
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=/usr/local/cuda:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/include:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=${TORCH}/lib/include:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

#nvcc --gpu-architecture=sm_52 -lineinfo -Xcompiler -rdynamic  -dc -odir src/ src/kernel_regression_kernel.cu \
#    --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC -L/usr/local/cuda/lib64/ -I/usr/local/cuda/include -lcublas -lcudadevrt -lcudart -lcublas_device -lm -rdc=true

nvcc -gencode=arch=compute_50,code=sm_50 \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_61,code=sm_61 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_70,code=compute_70 -Xcompiler '-fPIC' -dc -odir cffi/src/ cffi/src/kernel_regression_kernel.cu \
    --compiler-options -fPIC -I ${TORCH}/lib/include/TH -I ${TORCH}/lib/include/THC -L/usr/local/cuda/lib64/ -I/usr/local/cuda/include -lcublas -lcudadevrt -lcudart -lm -rdc=true
nvcc -gencode=arch=compute_50,code=sm_50 \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_60,code=sm_60 \
  -gencode=arch=compute_61,code=sm_61 \
  -gencode=arch=compute_70,code=sm_70 \
  -gencode=arch=compute_70,code=compute_70 -Xcompiler '-fPIC' -dlink cffi/src/kernel_regression_kernel.o --output-file cffi/src/kernel_regression_kernel_link.o -lcublas -lcudadevrt -lm
g++ -shared -o build/libkernel_regression_kernel_shared.so cffi/src/kernel_regression_kernel.o cffi/src/kernel_regression_kernel_link.o -fPIC -L/usr/local/cuda/lib64/ -lcudadevrt -lcudart -lcublas -lm

#nvcc -arch=sm_52 -lineinfo -Xcompiler -rdynamic  -dlink src/kernel_regression_kernel.o --output-file src/kernel_regression_kernel_link.o -lcublas -lcublas_device -lcudadevrt -lm
#g++ -shared -o build/libkernel_regression_kernel_shared.so src/kernel_regression_kernel.o src/kernel_regression_kernel_link.o -L/usr/local/cuda/lib64/ -lcudadevrt -lcudart -lcublas -lcublas_device -lm

python build_cuda.py

echo Done!
 
 
 

