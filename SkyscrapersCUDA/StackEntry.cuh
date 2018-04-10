#ifndef __INCLUDED_STACK_ENTRY_CUH__
#define __INCLUDED_STACK_ENTRY_CUH__

#include <limits>
#include "CUDAUtilities.cuh"

#define BIT_BASED_STACK

namespace cuda
{
    namespace solver
    {
        template <size_t size>
        class StackEntry
        {
        public:
            //CUDA_HOST StackEntry();
            //CUDA_HOST ~StackEntry();

            // Device constructors
            CUDA_DEVICE StackEntry();
            CUDA_DEVICE StackEntry(const StackEntry<size> & other);
            CUDA_DEVICE StackEntry(StackEntry<size> && other);
            CUDA_DEVICE ~StackEntry();

            // Device operators
            CUDA_DEVICE StackEntry<size>& operator=(StackEntry<size> other);
            CUDA_DEVICE StackEntry<size>& operator=(StackEntry<size> && other);
            CUDA_DEVICE void swap(StackEntry<size> & first, StackEntry<size> & second);

            // Constant due to CUDA limitations
            const size_t badIndex;

            // Bit info methods
            CUDA_DEVICE bool all() const;
            CUDA_DEVICE bool any() const;
            CUDA_DEVICE bool none() const;

            // Manipulation methods
            CUDA_DEVICE bool getBit(const size_t pos) const;
            CUDA_DEVICE void setBit(const size_t pos, bool val = true);
            CUDA_DEVICE void resetBit(const size_t pos);
            CUDA_DEVICE void flipBit(const size_t pos);

            CUDA_DEVICE size_t firstZero() const;
            
            CUDA_DEVICE size_t getSize() const;
            CUDA_HOST_DEVICE void clearAll();
        private:
#ifdef BIT_BASED_STACK
            typedef size_t stackLineT;
#else
            typedef size_t* stackLineT;
#endif // BIT_BASED_STACK
            size_t setBitsCount;
            stackLineT line;
        };

//        template <size_t size>
//        StackEntry<size>::StackEntry() : badIndex(std::numeric_limits<size_t>::max()), setBitsCount(0)
//        {
//#ifdef BIT_BASED_STACK
//            line = 0;
//#else
//            // Allocate memory for row
//            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&line), size * sizeof(*line));
//
//            // In case of error, print and reset pointer
//            if (err != cudaSuccess)
//            {
//                CUDA_PRINT_ERROR("Failed allocation", err);
//                line = nullptr;
//            }
//            else
//            {
//                clearAll();
//            }
//#endif // BIT_BASED_STACK
//        }
//
//        template <size_t size>
//        StackEntry<size>::~StackEntry()
//        {
//#ifdef BIT_BASED_STACK
//            // Nothing to do
//#else
//            cudaFree(line);
//            line = nullptr;
//            setBitsCount = 0;
//#endif // BIT_BASED_STACK
//        }

        template <size_t size>
        CUDA_DEVICE StackEntry<size>::StackEntry() : badIndex(~0), setBitsCount(0)
        {
#ifdef BIT_BASED_STACK
            line = 0;
#else
            // Allocate memory for row
            line = reinterpret_cast<stackLineT>(malloc(size * sizeof(*line)));

            // In case of error, print and reset pointer
            if (line == nullptr)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                line = nullptr;
            }
            else
            {
                clearAll();
            }
#endif // BIT_BASED_STACK
        }

        template<size_t size>
        CUDA_DEVICE StackEntry<size>::StackEntry(const StackEntry<size> & other) :
            badIndex(badIndex), setBitsCount(setBitsCount)
        {
#ifdef BIT_BASED_STACK
            line = other.line;
#else
            // Allocate memory for row
            line = reinterpret_cast<stackLineT>(malloc(size * sizeof(*line)));

            // In case of error, print and reset pointer
            if (line == nullptr)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                line = nullptr;
                setBitsCount = 0;
            }
            else
            {
                memcpy(line, other.line, size * sizeof(*line));
            }
#endif // BIT_BASED_STACK
        }

        template<size_t size>
        CUDA_DEVICE StackEntry<size>::StackEntry(StackEntry<size> && other) :
            badIndex(badIndex), setBitsCount(0), line(0)
        {
            swap(*this, other);
        }

        template <size_t size>
        CUDA_DEVICE StackEntry<size>::~StackEntry()
        {
#ifdef BIT_BASED_STACK
            // Nothing to do
#else
            free(line);
            line = nullptr;
            setBitsCount = 0;
#endif // BIT_BASED_STACK
        }

        template<size_t size>
        CUDA_DEVICE StackEntry<size> & StackEntry<size>::operator=(StackEntry<size> other)
        {
            swap(*this, other);
            return *this;
        }

        template<size_t size>
        CUDA_DEVICE StackEntry<size> & StackEntry<size>::operator=(StackEntry<size> && other)
        {
            swap(*this, other);
            return *this;
        }

        template<size_t size>
        CUDA_DEVICE void StackEntry<size>::swap(StackEntry<size> & first, StackEntry<size> & second)
        {
            size_t tmp_setBitsCount = first.setBitsCount;
            stackLineT tmp_line = first.line;
            first.setBitsCount = second.setBitsCount;
            first.line = second.line;
            second.setBitsCount = tmp_setBitsCount;
            second.line = tmp_line;
            
        }

        template <size_t size>
        CUDA_DEVICE bool StackEntry<size>::all() const
        {
            return getSize() == setBitsCount;
        }

        template <size_t size>
        CUDA_DEVICE bool StackEntry<size>::any() const
        {
            return setBitsCount > 0;
        }

        template <size_t size>
        CUDA_DEVICE bool StackEntry<size>::none() const
        {
            return setBitsCount == 0;
        }

        template <size_t size>
        CUDA_DEVICE bool StackEntry<size>::getBit(const size_t pos) const
        {
#ifdef BIT_BASED_STACK
            return !!((0x1 << pos) & line);
#else
            return line[pos];
#endif // BIT_BASED_STACK
        }

        template <size_t size>
        CUDA_DEVICE void StackEntry<size>::setBit(const size_t pos, bool val)
        {
#ifdef BIT_BASED_STACK
            size_t mask = static_cast<size_t>(val) << pos;

            if (getBit(pos) != val)
            {
                if (getBit(pos) == true)
                {
                    --setBitsCount;
                    line &= (~mask);
                }
                else
                {
                    ++setBitsCount;
                    line |= mask;
                }
            }
#else
            if (getBit(pos) != val)
            {
                if (getBit(pos) == true)
                {
                    --setBitsCount;
                }
                else
                {
                    ++setBitsCount;
                }
                line[pos] = val;
            }
#endif // BIT_BASED_STACK
        }

        template <size_t size>
        CUDA_DEVICE void StackEntry<size>::resetBit(const size_t pos)
        {
            setBit(pos, false);
        }

        template <size_t size>
        CUDA_DEVICE void StackEntry<size>::flipBit(const size_t pos)
        {
#ifdef BIT_BASED_STACK
            size_t mask = static_cast<size_t>(0x1) << pos;
            line ^= mask;
#else
            if (line[pos])
            {
                --setBitsCount;
            }
            else
            {
                ++setBitsCount;
            }
            line[pos] = !line[pos];
#endif // BIT_BASED_STACK

        }

        template <size_t size>
        CUDA_DEVICE size_t StackEntry<size>::firstZero() const
        {
#ifdef BIT_BASED_STACK
            const size_t bitsInLine = sizeof(line) * CHAR_BIT;
            decltype(line) mask = 1;

            size_t retVal = 0;
            for (; retVal < bitsInLine; retVal++, mask <<= 1)
            {
                if (line & mask)
                {
                    break;
                }
            }

            if (retVal == bitsInLine)
            {
                retVal = badIndex;
            }

            return retVal;
#else
            auto retVal = badIndex;
            if (!all())
            {
                size_t i = 0;
                for (; i < getSize(); i++)
                {
                    if (line[i] == 0)
                    {
                        break;
                    }
                }

                if (i < getSize())
                {
                    retVal = i;
                }
            }
            return retVal;
#endif // BIT_BASED_STACK

        }

        template <size_t size>
        CUDA_DEVICE size_t StackEntry<size>::getSize() const
        {
            return size;
        }

        template<size_t size>
        CUDA_HOST_DEVICE void StackEntry<size>::clearAll()
        {
#ifdef BIT_BASED_STACK
            line = 0;
#else
            cudaError_t err = cudaMemset(reinterpret_cast<void**>(&line), 0, size * sizeof(*line));
            // In case of error, print and free pointer
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memset", err);
                cudaFree(line);
                line = nullptr;
            }
#endif
        }
    }
}
#endif // !__INCLUDED_STACK_ENTRY_CUH__
