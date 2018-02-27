#ifndef __INCLUDED_BIT_MANIPULATION_CUH__
#define __INCLUDED_BIT_MANIPULATION_CUH__

#include "CUDAUtilities.cuh"

#define CUDA_BAD_INDEX (size_t(~0))

namespace cuda
{
    namespace solver
    {
        class BitManipulation
        {
        public:
            BitManipulation() = delete;
            ~BitManipulation() = delete;

            static CUDA_DEVICE bool all(size_t number);
            static CUDA_DEVICE bool any(size_t number);
            static CUDA_DEVICE bool none(size_t number);

            static CUDA_DEVICE bool getBit(const size_t & number, const size_t pos);
            static CUDA_DEVICE void setBit(size_t & number, const size_t pos, bool val = true);
            static CUDA_DEVICE void resetBit(size_t & number, const size_t pos);
            static CUDA_DEVICE void flipBit(size_t & number, const size_t pos);

            static CUDA_DEVICE size_t firstZero(const size_t & number);
        };
    }
}
#endif // !__INCLUDED_BIT_MANIPULATION_CUH__
