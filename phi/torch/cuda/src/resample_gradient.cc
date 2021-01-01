#include "helpers.h"
#include "torch/extension.h"

void LaunchResampleGradientKernel(
	const unsigned int dataBatchSize,
	const int dims,
	const unsigned int* __restrict__ dimSizes,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const unsigned int outputGradientSize,
	const float* __restrict__ outputGradient,
	const float* __restrict__ data,
	const float* __restrict__ points,
	float* __restrict__ dataGradient,
	float* __restrict__ pointsGradient,
	const Boundary* __restrict__ boundaries
);


torch::Tensor resample_gradient_op(const torch::Tensor outputGradient, const torch::Tensor data, const torch::Tensor points, const torch::Tensor boundaries, int grad_selection){
		// Prepare data access parameters
		assert(data.dim() >= 2);

		// dataBatchSize
		const unsigned int dataBatchSize = data.size(0);
		const unsigned int pointsBatchSize = points.size(0);
		assert(dataBatchSize == pointsBatchSize || dataBatchSize == 1 || pointsBatchSize == 1);
		//const unsigned int outputBatchSize = outputGradient.shape().dim_size(0);
		unsigned int outputBatchSize = dataBatchSize > pointsBatchSize ? dataBatchSize : pointsBatchSize;
		// dims
		const int dims = data.dim() - 2;
		assert(dims == points.size(points.dim() - 1));
		assert(dims == boundaries.size(0) && boundaries.size(1) == 2);
		// dimSizes
		unsigned int dimSizes[dims];
		for(int i = 0; i < dims; i++){
			dimSizes[i] = data.size(i + 1);
		}
		// components
		const unsigned int components = data.size(data.dim() - 1);
		// pointsSize
		const unsigned int pointsSize = points.numel();

		// Create output tensors
		torch::Tensor dataGradient = torch::zeros_like(data, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
		torch::Tensor pointsGradient = torch::zeros_like(points, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

        //outputSize
        unsigned int outputSize = 1;
        for(int i = 1; i < points.dim() - 1; i++) {
            outputSize *= points.size(i);
        }
        outputSize *= outputBatchSize * components;
        //unsigned int outputSize = outputGradient.NumElements() * outputBatchSize;

		// outputElementsPerBatch
		const unsigned int outputElementsPerBatch = outputSize / outputBatchSize;

		// Do the computation.
		LaunchResampleGradientKernel(
			dataBatchSize,
			dims,
			dimSizes,
			components,
			pointsSize,
			outputElementsPerBatch,
			outputSize,
			outputGradient.numel(),
			outputGradient.data_ptr<float>(),
			data.data_ptr<float>(),
			points.data_ptr<float>(),
			dataGradient.data_ptr<float>(),
			pointsGradient.data_ptr<float>(),
			(Boundary*) boundaries.data_ptr<int>()
		);

	if(grad_selection == 1){
	    return dataGradient;
	}
	else if (grad_selection == 2){
	    return pointsGradient;
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("resample_gradient_op", &resample_gradient_op, "PyTorch CUDA Resampling Gradient");
}