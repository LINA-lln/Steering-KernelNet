import os
import glob
import torch.utils.ffi

strBasepath = os.path.split(os.path.abspath(__file__))[0] + '/'


# kernel_regression 
strHeaders = ['cffi/include/kernel_regression_cuda.h']
strSources = ['cffi/src/kernel_regression_cuda.c']
strDefines = [('WITH_CUDA', None)]
strObjects = ['build/libkernel_regression_kernel_shared.so']
strObjects += glob.glob('/usr/local/cuda/lib64/*.a')
strIncludes = ['cffi/include']


ffi = torch.utils.ffi.create_extension(
    name='_ext.kernel_regression_cuda',
    headers=strHeaders,
    sources=strSources,
    verbose=False,
    with_cuda=True,
    package=False,
    relative_to=strBasepath,
    extra_compile_args=["-std=c99"],
    # include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
    define_macros=strDefines,
    extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects],
    include_dirs=[os.path.join(strBasepath, strInclude) for strInclude in strIncludes]
)

if __name__ == '__main__':
#    assert( torch.cuda.is_available() == True)
    ffi.build()

 


