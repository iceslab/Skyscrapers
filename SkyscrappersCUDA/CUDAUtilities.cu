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

        size_t fifoSize = 0;
        cudaDeviceGetLimit(&fifoSize, cudaLimitPrintfFifoSize);
        auto converted = bytesToHumanReadable(fifoSize);
        fprintf(stderr, "FIFO size (printf): %5.1f %s\n", converted.first, converted.second.c_str());
        fifoSize = (512 << 20);
        converted = bytesToHumanReadable(fifoSize);
        fprintf(stderr, "Setting FIFO size to %5.1f %s\n", converted.first, converted.second.c_str());
        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, fifoSize);
        cudaDeviceGetLimit(&fifoSize, cudaLimitPrintfFifoSize);
        converted = bytesToHumanReadable(fifoSize);
        fprintf(stderr, "FIFO size (printf): %5.1f %s\n", converted.first, converted.second.c_str());

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