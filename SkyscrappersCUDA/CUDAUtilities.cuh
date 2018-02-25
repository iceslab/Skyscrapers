#ifndef __INCLUDED_CUDA_UTILITIES_CUH__
#define __INCLUDED_CUDA_UTILITIES_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

//#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_GLOBAL __global__
//#else
//#define CUDA_HOST
//#define CUDA_DEVICE
//#define CUDA_HOST_DEVICE
//#define CUDA_GLOBAL
//#endif 

#define CUDA_PRINT_ERROR(description, errorCode) \
do{ \
    fprintf(stderr, "%s %s: %s\n", __FUNCSIG__, description, cudaGetErrorString(errorCode)); \
} while (false);

#define CUDA_PRINT(...) \
do{ \
    printf(__VA_ARGS__); \
} while (false);

#define HOST_PRINT_ERROR(description) \
do{ \
    fprintf(stderr, "%s %s\n", __FUNCSIG__, description); \
} while (false);

#define CUDA_SOFT_ASSERT(expr) \
do{ \
    if(!(expr)) \
    { \
        fprintf(stderr, "Assertion failed: " QUOTE(expr) ", file %s, line %d\n", __FILE__, __LINE__); \
    } \
} while (false);

#define CUDA_SOFT_ASSERT_VERBOSE(expr, format, ...) \
do{ \
    if(!(expr)) \
    { \
        fprintf(stderr, "Assertion failed: " QUOTE(expr) ", "); \
        fprintf(stderr, format, __VA_ARGS__); \
        fprintf(stderr, ", file %s, line %d\n", __FILE__, __LINE__); \
    } \
} while (false);

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(P) (P)
#endif //!UNREFERENCED_PARAMETER

namespace cuda
{
    cudaError_t initDevice();
    cudaError_t deinitDevice();
}

#endif // !__INCLUDED_CUDA_UTILITIES_CUH__
