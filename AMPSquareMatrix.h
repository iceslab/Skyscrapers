#pragma once
#include <amp.h>

namespace AMP
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

    template<class T> class AMPSquareMatrix : public Concurrency::array<T, 2>
    {
    public:
        typedef Concurrency::array_view<T, 1> rowT;
        typedef Concurrency::array_view<T, 1> columnConstT;
        typedef Concurrency::array_view<T, 1> columnT;

        AMPSquareMatrix(const size_t size = 0) __CPU_ONLY;
        AMPSquareMatrix(const AMPSquareMatrix & matrix) __CPU_ONLY;
        ~AMPSquareMatrix() __CPU_ONLY;

        /// Accessors

        const rowT& getRow(size_t index) const __GPU;
        rowT& getRow(size_t index) __GPU;

        columnConstT getColumn(size_t index) const __GPU;
        columnT getColumn(size_t index) __GPU;

        const T getCell(size_t row, size_t column) const __GPU;
        T getCell(size_t row, size_t column) __GPU;

        void setCell(size_t row, size_t column, T value) __GPU;

        size_t size() const __GPU;

        SideE whichEdgeRow(size_t row) const __GPU;
        SideE whichEdgeColumn(size_t column) const __GPU;
        void fill(const T & value) __GPU;
    };

    template<class T>
    inline AMPSquareMatrix<T>::AMPSquareMatrix(const size_t size) :
        Concurrency::array<T, 2>(size, size) __CPU_ONLY
    {
        // Nothing to do
    }

    template<class T>
    inline AMPSquareMatrix<T>::AMPSquareMatrix(const AMPSquareMatrix & matrix) __CPU_ONLY
    {
        matrix.copy_to(*this);
    }

    template<class T>
    inline AMPSquareMatrix<T>::~AMPSquareMatrix() __CPU_ONLY
    {
        // Nothing to do
    }

    template<class T>
    inline const typename AMPSquareMatrix<T>::rowT & AMPSquareMatrix<T>::getRow(size_t index) const __GPU
    {
        return (*this)[index];
    }

    template<class T>
    inline typename AMPSquareMatrix<T>::rowT & AMPSquareMatrix<T>::getRow(size_t index) __GPU
    {
        return (*this)[index];
    }

    template<class T>
    inline typename AMPSquareMatrix<T>::columnConstT AMPSquareMatrix<T>::getColumn(size_t index) const __GPU
    {
        columnConstT column;
        column.reserve(size());

        for (auto& row : *this)
        {
            column.push_back(row[index]);
        }

        return column;
    }

    template<class T>
    inline typename AMPSquareMatrix<T>::columnT AMPSquareMatrix<T>::getColumn(size_t index) __GPU
    {
        columnT column;
        column.reserve(size());

        for (auto& row : *this)
        {
            column.push_back(row[index]);
        }

        return column;
    }

    template<class T>
    inline const T AMPSquareMatrix<T>::getCell(size_t row, size_t column) const __GPU
    {
        return (*this)[row][column];
    }

    template<class T>
    inline T AMPSquareMatrix<T>::getCell(size_t row, size_t column) __GPU
    {
        return (*this)[row][column];
    }

    template<class T>
    inline void AMPSquareMatrix<T>::setCell(size_t row, size_t column, T value) __GPU
    {
        (*this)[row][column] = value;
    }

    template<class T>
    inline size_t AMPSquareMatrix<T>::size() const
    {
        return get_extent()[0];
    }

    template<class T>
    inline SideE AMPSquareMatrix<T>::whichEdgeRow(size_t row) const __GPU
    {
        if (row == 0)
        {
            return TOP;
        }
        else if (row == (size() - 1))
        {
            return BOTTOM;
        }
        else
        {
            return NONE;
        }
    }

    template<class T>
    inline SideE AMPSquareMatrix<T>::whichEdgeColumn(size_t column) const __GPU
    {
        if (column == 0)
        {
            return LEFT;
        }
        else if (column == (size() - 1))
        {
            return RIGHT;
        }
        else
        {
            return NONE;
        }
    }

    template<class T>
    inline void AMPSquareMatrix<T>::fill(const T & value) __GPU
    {
        for (size_t row = 0; row < size(); row++)
        {
            for (size_t column = 0; column < size(); column++)
            {
                setCell(row, column, value);
            }
        }
    }
}
