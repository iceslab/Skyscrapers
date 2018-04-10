#include "BitManipulation.cuh"

namespace cuda
{
    namespace solver
    {
        CUDA_DEVICE bool BitManipulation::all(uint32T number)
        {
            return (~number) == 0;
        }

        CUDA_DEVICE bool BitManipulation::any(uint32T number)
        {
            return number != 0;
        }

        CUDA_DEVICE bool BitManipulation::none(uint32T number)
        {
            return number == 0;
        }

        CUDA_DEVICE bool BitManipulation::getBit(const uint32T & number, const uint32T pos)
        {
            return !!((0x1 << pos) & number);
        }

        CUDA_DEVICE void BitManipulation::setBit(uint32T & number, const uint32T pos, bool val)
        {
            uint32T mask = static_cast<uint32T>(val) << pos;

            if (getBit(number, pos) != val)
            {
                if (getBit(number, pos) == true)
                {
                    number &= (~mask);
                }
                else
                {
                    number |= mask;
                }
            }
        }

        CUDA_DEVICE void BitManipulation::resetBit(uint32T & number, const uint32T pos)
        {
            setBit(number, pos, false);
        }

        CUDA_DEVICE void BitManipulation::flipBit(uint32T & number, const uint32T pos)
        {
            uint32T mask = static_cast<uint32T>(0x1) << pos;
            number ^= mask;
        }

        CUDA_DEVICE uint32T BitManipulation::firstZero(const uint32T & number)
        {
#ifdef USE_INTEGER_INTRINSICS

            // Casting explicitly to int32T to match argument type
            // Negating, because it counts position of least significant bit set to 1
            // It means, desired outcome will be: result - 1 (first bit has index 1)
            // If all bits are set to one (and after negating to 0) it'll return 0
            uint32T retVal = static_cast<uint32T>(__ffs(static_cast<int32T>(~number)));

            --retVal;
            // Due to CUDA_BAD_INDEX defined as negation of 0
            // decrementing is equivalent to this code:
            //
            // if (retVal == 0)
            // {
            //     retVal = CUDA_BAD_INDEX;
            // }
            // else
            // {
            //     --retVal;
            // }
            //

            // Old implementation
#else
            const uint32T bitsInNumber = sizeof(number) * CHAR_BIT;
            uint32T mask = 1;

            //CUDA_PRINT("%s: number=0x%08llx, mask=0x%08llx, bits=%llu\n", __FUNCTION__, number, mask, bitsInNumber);
            size_t retVal = 0;
            for (; retVal < bitsInNumber; retVal++, mask <<= 1)
            {
                if (!(number & mask))
                {
                    break;
                }
            }

            if (retVal == bitsInNumber)
            {
                retVal = CUDA_BAD_INDEX;
            }
#endif

            return retVal;
        }
    }
}
