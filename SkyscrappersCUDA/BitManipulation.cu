#include "BitManipulation.cuh"

namespace cuda
{
    namespace solver
    {
        CUDA_DEVICE bool BitManipulation::all(size_t number)
        {
            return (~number) == 0;
        }

        CUDA_DEVICE bool BitManipulation::any(size_t number)
        {
            return number != 0;
        }

        CUDA_DEVICE bool BitManipulation::none(size_t number)
        {
            return number == 0;
        }

        CUDA_DEVICE bool BitManipulation::getBit(const size_t & number, const size_t pos)
        {
            return !!((0x1 << pos) & number);
        }

        CUDA_DEVICE void BitManipulation::setBit(size_t & number, const size_t pos, bool val)
        {
            size_t mask = static_cast<size_t>(val) << pos;

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

        CUDA_DEVICE void BitManipulation::resetBit(size_t & number, const size_t pos)
        {
            setBit(number, pos, false);
        }

        CUDA_DEVICE void BitManipulation::flipBit(size_t & number, const size_t pos)
        {
            size_t mask = static_cast<size_t>(0x1) << pos;
            number ^= mask;
        }

        CUDA_DEVICE size_t BitManipulation::firstZero(const size_t & number)
        {
            const size_t bitsInNumber = sizeof(number) * CHAR_BIT;
            size_t mask = 1;
            
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

            return retVal;
        }
    }
}
