#include "CUDAUtilities.cuh"

namespace cuda
{
    cudaError_t initDevice()
    {
        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaError_t cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }

        return cudaStatus;
    }

    cudaError_t deinitDevice()
    {
        cudaError_t cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceReset failed!");
        }

        return cudaStatus;
    }

}