#ifndef __INCLUDED_SQUARE_MATRIX_H__
#define __INCLUDED_SQUARE_MATRIX_H__

#include "CUDAUtilities.cuh"

namespace cuda
{
    // Enum for accessing hints array
    enum SideE
    {
        NONE = -1, // For indication that it is not on the edge
        TOP = 0,
        RIGHT,
        BOTTOM,
        LEFT
    };

    template<class T> class SquareMatrix
    {
    public:
        typedef T * rowT;
        typedef const T * columnConstT;
        typedef T * columnT;
        typedef T * setIntersectionT;

        //SquareMatrix(const SquareMatrix & matrix);
        //SquareMatrix(SquareMatrix && matrix);

        SquareMatrix & operator=(const SquareMatrix & matrix);
        SquareMatrix & operator=(SquareMatrix && matrix);

        SquareMatrix(const size_t size);
        ~SquareMatrix();

        /// Accessors

        CUDA_DEVICE rowT getRow(size_t index) const;

        CUDA_DEVICE columnT getColumn(size_t index) const;

        CUDA_DEVICE const T getCell(size_t row, size_t column) const;

        CUDA_DEVICE void setCell(size_t row, size_t column, T value);

        CUDA_HOST_DEVICE size_t getSize() const;

        /// Helper methods

        CUDA_HOST_DEVICE SideE whichEdgeRow(size_t row) const;
        CUDA_HOST_DEVICE SideE whichEdgeColumn(size_t column) const;

        CUDA_DEVICE void fill(const T & value);
        CUDA_HOST void clear();

        //CUDA_HOST void deduceSetValues();

    protected:
        T * d_data;
        size_t size;
    };

    template<class T>
    inline SquareMatrix<T>::SquareMatrix(const size_t size)
    {
        this->size = size;
        if(size > 0)
        {
            const cudaError_t err = cudaMalloc(&d_data, size * size * sizeof(T));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                this->size = 0;
                d_data = nullptr;
            }
            else
            {
                clear();
            }
        }
    }

    template<class T>
    inline SquareMatrix<T>::~SquareMatrix()
    {
        cudaFree(d_data);
        d_data = nullptr;
        size = 0;
    }

    template<class T>
    inline CUDA_DEVICE typename SquareMatrix<T>::rowT SquareMatrix<T>::getRow(size_t index) const
    {
        rowT retVal = d_data + index * size;
        return retVal;
    }

    template<class T>
    inline CUDA_DEVICE typename SquareMatrix<T>::columnT SquareMatrix<T>::getColumn(size_t index) const
    {
        T* retVal = nullptr;

        // Allocate memory for column
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&retVal), size * sizeof(T));

        // In case of error, print and reset pointer
        if (err != cudaSuccess)
        {
            CUDA_PRINT_ERROR("Failed allocation", err);
            retVal = nullptr;
        }
        else
        {
            // Copy respective column to newly allocated memory
            for (size_t i = 0; i < size; i++)
            {
                retVal[i] = d_data[i * size + index];
            }
        }
        return retVal;
    }

    template<class T>
    inline CUDA_DEVICE const T SquareMatrix<T>::getCell(size_t row, size_t column) const
    {
        return d_data[row * size + column];
    }

    template<class T>
    inline CUDA_DEVICE void SquareMatrix<T>::setCell(size_t row, size_t column, T value)
    {
        d_data[row * size + column] = value;
    }

    template<class T>
    inline CUDA_HOST_DEVICE size_t SquareMatrix<T>::getSize() const
    {
        return size;
    }

    template<class T>
    inline CUDA_HOST_DEVICE SideE SquareMatrix<T>::whichEdgeRow(size_t row) const
    {
        if (row == 0)
        {
            return TOP;
        }
        else if (row == (size - 1))
        {
            return BOTTOM;
        }
        else
        {
            return NONE;
        }
    }

    template<class T>
    inline CUDA_HOST_DEVICE SideE SquareMatrix<T>::whichEdgeColumn(size_t column) const
    {
        if (column == 0)
        {
            return LEFT;
        }
        else if (column == (size - 1))
        {
            return RIGHT;
        }
        else
        {
            return NONE;
        }
    }

    template<class T>
    inline CUDA_DEVICE void SquareMatrix<T>::fill(const T & value)
    {
        T* const beginIt = d_data;
        T* const endIt = beginIt + size * size * sizeof(T);

        for (T* it = beginIt; it < endIt; it++)
        {
            *it = value;
        }
    }

    template<class T>
    inline CUDA_HOST void SquareMatrix<T>::clear()
    {
        cudaError_t err = cudaMemset(reinterpret_cast<void*>(d_data), 0, size * size * sizeof(T));
        if (err != cudaSuccess)
        {
            CUDA_PRINT_ERROR("Failed memset", err);
        }
    }
};

#endif // !__INCLUDED_SQUARE_MATRIX_H__
