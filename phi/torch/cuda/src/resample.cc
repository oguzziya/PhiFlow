#include "helpers.h"
#include <cassert>
#include "torch/extension.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA Tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace torch_cuda {

void LaunchResampleKernel(
        const unsigned int dataBatchSize,
        const int dims,
        const unsigned int* __restrict__ dimSizes,
        const unsigned int components,
        const unsigned int pointsSize,
        const unsigned int outputElementsPerBatch,
        const unsigned int outputSize,
        const float* __restrict__ data,
        const float* __restrict__ points,
        float* __restrict__ output,
        const Boundary* __restrict__ boundaries
);

torch::Tensor resample_op(torch::Tensor data, torch::Tensor points, const torch::Tensor boundaries)   {

    CHECK_INPUT(data);
    CHECK_INPUT(points);
    CHECK_INPUT(boundaries);

    // Prepare data access parameters
    assert(data.dim() >= 2);
    // dataBatchSize
    const unsigned int dataBatchSize = data.size(0);
    const unsigned int pointsBatchSize = points.size(0);
    assert(dataBatchSize == pointsBatchSize || dataBatchSize == 1 || pointsBatchSize == 1);
    const unsigned int outputBatchSize = dataBatchSize > pointsBatchSize ? dataBatchSize : pointsBatchSize;
    // dims
    const int dims = data.dim() - 2;
    assert(dims == points.size(points.dim() - 1));
    assert(dims == boundaries.size(0) && boundaries.size(1) == 2);
    // dimSizes
    unsigned int dimSizes[dims];
    std::vector<int> outputDims(dims+2);
    outputDims[0] = 1;
    outputDims[dims-1] = 1;
    for(int i = 0; i < dims; i++){
        dimSizes[i] = data.size(i + 1);
    }
    for(int i = 1; i < dims+1; i++){
        outputDims[i] = points.size(i);
    }
    // components
    const unsigned int components = data.size(data.dim() - 1);
    // pointsSize
    const unsigned int pointsSize = points.numel();

    torch::Tensor outputTensor = torch::zeros({1, outputDims[1], outputDims[2], 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    // outputElementsPerBatch
    const unsigned int outputElementsPerBatch = outputTensor.numel() / outputBatchSize;

    LaunchResampleKernel(
            dataBatchSize,
            dims,
            dimSizes,
            components,
            pointsSize,
            outputElementsPerBatch,
            outputTensor.numel(),
            data.data_ptr<float>(),
            points.data_ptr<float>(),
            outputTensor.data_ptr<float>(),
            (Boundary*) boundaries.data_ptr<int>()
            );

    return outputTensor;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("resample_op", &torch_cuda::resample_op, "PyTorch CUDA Resampling");
}