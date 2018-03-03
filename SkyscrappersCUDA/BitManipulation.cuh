#ifndef __INCLUDED_BIT_MANIPULATION_CUH__
#define __INCLUDED_BIT_MANIPULATION_CUH__

#include "CUDAUtilities.cuh"

#define USE_INTEGER_INTRINSICS
#define CUDA_BAD_INDEX (cuda::uint32T(~0))

namespace cuda
{
    namespace solver
    {
        class BitManipulation
        {
        public:
            BitManipulation() = delete;
            ~BitManipulation() = delete;

            static CUDA_DEVICE bool all(uint32T number);
            static CUDA_DEVICE bool any(uint32T number);
            static CUDA_DEVICE bool none(uint32T number);

            static CUDA_DEVICE bool getBit(const uint32T & number, const uint32T pos);
            static CUDA_DEVICE void setBit(uint32T & number, const uint32T pos, bool val = true);
            static CUDA_DEVICE void resetBit(uint32T & number, const uint32T pos);
            static CUDA_DEVICE void flipBit(uint32T & number, const uint32T pos);

            static CUDA_DEVICE uint32T firstZero(const uint32T & number);
        };
    }
}
#endif // !__INCLUDED_BIT_MANIPULATION_CUH__
