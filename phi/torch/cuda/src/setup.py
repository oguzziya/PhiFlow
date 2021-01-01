from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='resample_torch',
    ext_modules=[
        CUDAExtension('resample_torch_cuda', ['resample_kernel.cu', 'resample.cc'],
                      extra_compile_args={'cxx':['-D_GLIBCXX_USE_CXX11_ABI=0'],
                                          'nvcc':[]}),
        CUDAExtension('resample_torch_gradient_cuda', ['resample_gradient_kernel.cu', 'resample_gradient.cc'],
                      extra_compile_args={'cxx':['-D_GLIBCXX_USE_CXX11_ABI=0'],
                                          'nvcc':[]})
    ],
    cmdclass={'build_ext': BuildExtension}
)