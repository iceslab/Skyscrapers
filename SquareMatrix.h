#pragma once
#include <vector>
namespace matrix
{
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
}