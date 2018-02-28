#include "CUDAUtilities.cuh"

size_t desiredFifoSize = CUDA_DEFAULT_FIFO_SIZE;

namespace cuda
{
    cudaError_t initDevice(size_t fifoSize)
    {
        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaError_t cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }

        if (fifoSize != CUDA_DEFAULT_FIFO_SIZE)
        {
            size_t fifoSizeRef = 0;
            cudaDeviceGetLimit(&fifoSizeRef, cudaLimitPrintfFifoSize);
            auto converted = bytesToHumanReadable(fifoSizeRef);
            fprintf(stderr, "FIFO size (printf): %5.1f %s\n", converted.first, converted.second.c_str());
            converted = bytesToHumanReadable(fifoSize);
            fprintf(stderr, "Setting FIFO size to %5.1f %s\n", converted.first, converted.second.c_str());
            cudaDeviceSetLimit(cudaLimitPrintfFifoSize, fifoSize);
            cudaDeviceGetLimit(&fifoSizeRef, cudaLimitPrintfFifoSize);
            converted = bytesToHumanReadable(fifoSizeRef);
            fprintf(stderr, "FIFO size (printf): %5.1f %s\n", converted.first, converted.second.c_str());
        }
        else
        {
            auto converted = bytesToHumanReadable(fifoSize);
            fprintf(stderr, "Default FIFO size (printf): %5.1f %s\n", converted.first, converted.second.c_str());
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

    std::pair<double, std::string> bytesToHumanReadable(double bytes)
    {
        const std::vector<std::string> postfixes = {"B", "KB", "MB", "GB", "TB", "PB", "EB"};
        const double factor = 1024.0;

        size_t i = 0;
        for (; i < postfixes.size(); i++)
        {
            if (bytes < factor)
            {
                break;
            }
            bytes /= factor;
        }

        return std::make_pair(bytes, postfixes[i]);
    }

}