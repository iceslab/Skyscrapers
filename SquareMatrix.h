#pragma once
#include <vector>
namespace matrix
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

    template<class T> class SquareMatrix : public std::vector<std::vector<T>>
    {
    public:
        typedef std::vector<T> rowT;
        typedef std::vector<std::reference_wrapper<const T>> columnConstT;
        typedef std::vector<std::reference_wrapper<T>> columnT;
        typedef std::vector<T> setIntersectionT;

        SquareMatrix(const size_t size);
        ~SquareMatrix() = default;

        // Accessors

        const rowT& getRow(size_t index) const;
        rowT& getRow(size_t index);

        columnConstT getColumn(size_t index)  const;
        columnT getColumn(size_t index);

        // Helper methods

        SideE whichEdgeRow(size_t row);
        SideE whichEdgeColumn(size_t column);

        void fill(const T & value);
    };

    template<class T>
    inline SquareMatrix<T>::SquareMatrix(const size_t size) : std::vector<std::vector<T>>(size, rowT(size))
    {
        // Nothing to do
    }

    template<class T>
    inline const typename SquareMatrix<T>::rowT & SquareMatrix<T>::getRow(size_t index) const
    {
        return (*this)[index];
    }

    template<class T>
    inline typename SquareMatrix<T>::rowT & SquareMatrix<T>::getRow(size_t index)
    {
        return (*this)[index];
    }

    template<class T>
    inline typename SquareMatrix<T>::columnConstT SquareMatrix<T>::getColumn(size_t index) const
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
    inline typename SquareMatrix<T>::columnT SquareMatrix<T>::getColumn(size_t index)
    {
        columnT column;
        column.reserve(getSize());

        for (auto& row : *this)
        {
            column.push_back(row[index]);
        }

        return column;
    }

    template<class T>
    inline SideE SquareMatrix<T>::whichEdgeRow(size_t row)
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
    inline SideE SquareMatrix<T>::whichEdgeColumn(size_t column)
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
    inline void SquareMatrix<T>::fill(const T & value)
    {
        for (auto& row : (*this))
        {
            std::fill(row.begin(), row.end(), value);
        }
    }
}