#ifndef __INCLUDED_CUDA_UTILITIES_CUH__
#define __INCLUDED_CUDA_UTILITIES_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <tuple>
#include <string>
#include <vector>

//#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_GLOBAL __global__
#define CUDA_SHARED __shared__
#define CUDA_CONSTANT __constant__

#define CUDA_DEFAULT_FIFO_SIZE (1 << 20) // default FIFO size in bytes

namespace cuda
{
    typedef signed int int32T;
    typedef unsigned int uint32T;
    typedef signed long long int int64T;
    typedef unsigned long long int uint64T;

    typedef struct cudaEventsDevice
    {
        int64T initBegin;
        int64T initEnd;
        int64T loopBegin;
        int64T loopEnd;

        int64T firstZeroDiff;
        int64T goodIndexDiff;
        int64T badIndexDiff;
        int64T placeableDiff;
        int64T placeableFnDiff;
        int64T boardValidDiff;
        int64T boardInvalidDiff;
        int64T boardValidFnDiff;
        int64T lastCellDiff;
        int64T notLastCellDiff;
        int64T copyResultDiff;
    } cudaEventsDeviceT;

    enum Resolution
    {
        SECONDS = 1,
        MILLISECONDS = SECONDS * 1000,
        MICROSECONDS = MILLISECONDS * 1000,
        NANOSECONDS = MICROSECONDS * 1000
    };
}

#define CUDA_UINT32T_MAX (cuda::uint32T(~0))
#define CUDA_MAX_SHARED_MEMORY (32 << 10) // 32 kB

//#else
//#define CUDA_HOST
//#define CUDA_DEVICE
//#define CUDA_HOST_DEVICE
//#define CUDA_GLOBAL
//#endif 

#define CUDA_PRINT_ERROR(description, errorCode) \
do{ \
    printf("%s %s: %s\n", __FUNCSIG__, description, cudaGetErrorString(errorCode)); \
} while (false);

#define CUDA_PRINT(...) \
do{ \
    printf(__VA_ARGS__); \
} while (false);

#define HOST_PRINT_ERROR(description) \
do{ \
    printf("%s %s\n", __FUNCSIG__, description); \
} while (false);

#define CUDA_SOFT_ASSERT(expr) \
do{ \
    if(!(expr)) \
    { \
        printf("Assertion failed: " QUOTE(expr) ", file %s, line %d\n", __FILE__, __LINE__); \
    } \
} while (false);

#define CUDA_SOFT_ASSERT_VERBOSE(expr, format, ...) \
do{ \
    if(!(expr)) \
    { \
        printf("Assertion failed: " QUOTE(expr) ", "); \
        printf(format, __VA_ARGS__); \
        printf(", file %s, line %d\n", __FILE__, __LINE__); \
    } \
} while (false);

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(P) (P)
#endif //!UNREFERENCED_PARAMETER

namespace cuda
{
    cudaError_t initDevice(size_t fifoSize = CUDA_DEFAULT_FIFO_SIZE);
    cudaError_t deinitDevice();

    std::pair<double, std::string> bytesToHumanReadable(double bytes);
    std::pair<double, std::string> timeToHumanReadable(double time, Resolution resolution);
    double getTime(int64T start, int64T end, Resolution resolution = SECONDS);
    double getTime(int64T diff, Resolution resolution = SECONDS);
}

#endif // !__INCLUDED_CUDA_UTILITIES_CUH__
