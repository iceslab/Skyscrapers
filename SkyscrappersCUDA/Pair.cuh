#ifndef __INCLUDED_PAIR_CUH__
#define __INCLUDED_PAIR_CUH__

#include "CUDAUtilities.cuh"

namespace cuda
{
    template <typename T, typename U>
    struct Pair
    {
        CUDA_HOST_DEVICE Pair(T first, U second);
        ~Pair() = default;

        CUDA_HOST_DEVICE Pair & operator=(const Pair & other);

        CUDA_HOST_DEVICE bool operator==(const Pair & other) const;
        CUDA_HOST_DEVICE bool operator!=(const Pair & other) const;
        T first;
        U second;
    };

    template<typename T, typename U>
    CUDA_HOST_DEVICE Pair<T, U>::Pair(T first, U second) : first(first), second(second)
    {
        // Nothing to do
    }

    template<typename T, typename U>
    CUDA_HOST_DEVICE Pair<T, U> & Pair<T, U>::operator=(const Pair & other)
    {
        first = other.first;
        second = other.second;
        return *this;
    }

    template<typename T, typename U>
    CUDA_HOST_DEVICE bool Pair<T, U>::operator==(const Pair & other) const
    {
        return first == other.first && second == other.second;
    }

    template<typename T, typename U>
    CUDA_HOST_DEVICE bool Pair<T, U>::operator!=(const Pair & other) const
    {
        return !((*this) == other);
    }
}

#endif // !__INCLUDED_PAIR_CUH__
