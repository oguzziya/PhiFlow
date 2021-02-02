#include "helpers.h"
#include <cassert>
#include "torch/extension.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA Tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void LaunchPressureKernel(const int *dimensions, const int dim_product, const int dimSize,
            const signed char* laplace_matrix,
            float* p, float* z, float* r, float *divergence, float* x,
            const float* oneVector,
            bool* thresholdReached,
            const float accuracy,
            const int max_iterations,
            const int batch_size,
            int* iterations_gpu);

void LaplaceMatrixKernelLauncher(const int *dimensions, const int dimSize, const int dim_product, 
            const float *active_mask, const float *fluid_mask, const int *maskDimensions, 
            signed char *laplace_matrix, int *cords);

void pressure_solve_op(torch::Tensor dimensions, torch::Tensor mask_dimensions, torch::Tensor active_mask, torch::Tensor fluid_mask,
                    torch::Tensor laplace_matrix, torch::Tensor divergence, torch::Tensor p, torch::Tensor r,
                    torch::Tensor z, torch::Tensor pressure, torch::Tensor one_vector, int dim_product, float accuracy, int max_iterations) {
    auto begin = std::chrono::high_resolution_clock::now();

    int batch_size = divergence.size(0);
    int dim_size = dimensions.dim();

    auto end = std::chrono::high_resolution_clock::now();

//        printf("General Preparation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);

    begin = std::chrono::high_resolution_clock::now();
    // Laplace:
    // Laplace Helper
    torch::Tensor cords = torch::zeros({dim_product, dim_size}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    end = std::chrono::high_resolution_clock::now();

//        printf("Laplace Preparation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);


    begin = std::chrono::high_resolution_clock::now();
    LaplaceMatrixKernelLauncher(dimensions.data_ptr<int>(), dim_size, dim_product, active_mask.data_ptr<float>(),
                                fluid_mask.data_ptr<float>(), mask_dimensions.data_ptr<int>(),
                                laplace_matrix.data_ptr<signed char>(), cords.data_ptr<int>());
    end = std::chrono::high_resolution_clock::now();

//        printf("Laplace Matrix Generation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);


    begin = std::chrono::high_resolution_clock::now();

    torch::Tensor threshold_reached = torch::zeros({batch_size, dim_size}, torch::dtype(torch::kBool).device(torch::kCUDA, 0));
    auto iterations = torch::zeros({1}, (torch::dtype(torch::kInt8).device(torch::kCUDA, 0)));

    end = std::chrono::high_resolution_clock::now();

//        printf("Pressure Solve Preparation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);

    begin = std::chrono::high_resolution_clock::now();
    LaunchPressureKernel(dimensions.data_ptr<int>(), dim_product, dim_size,
                          laplace_matrix.data_ptr<signed char>(),
                          p.data_ptr<float>(), z.data_ptr<float>(), r.data_ptr<float>(), divergence.data_ptr<float>(), pressure.data_ptr<float>(),
                          one_vector.data_ptr<float>(),
                          threshold_reached.data_ptr<bool>(),
                          accuracy,
                          max_iterations,
                          batch_size,
                          iterations.data_ptr<int>());
    end = std::chrono::high_resolution_clock::now();


//        printf("Pressure Solve took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);
//        printf("%f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("pressure_solve_op", &pressure_solve_op, "PyTorch CUDA Pressure Poisson Equation Solver");
}
