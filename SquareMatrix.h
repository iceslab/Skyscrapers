#pragma once
#include <vector>
#include <functional>
#include <iterator>
#include <fstream>

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

        SquareMatrix(const size_t size = 0);
        ~SquareMatrix() = default;

        bool saveToFile(const std::string & path) const;
        bool saveToFile(std::ofstream & stream) const;

        bool readFromFile(const std::string & path);
        bool readFromFile(std::ifstream & stream);

        /// Accessors

        const rowT& getRow(size_t index) const;
        rowT& getRow(size_t index);

        columnConstT getColumn(size_t index)  const;
        columnT getColumn(size_t index);

        const T getCell(size_t row, size_t column) const;
        T getCell(size_t row, size_t column);
        
        void setCell(size_t row, size_t column, T value);

        /// Helper methods

        SideE whichEdgeRow(size_t row) const;
        SideE whichEdgeColumn(size_t column) const;

        void forEachCell(std::function<void(size_t, size_t)> function);
        void forEachVector(std::function<void(size_t, size_t)> function);

        void fill(const T & value);
    };

    template<class T>
    inline SquareMatrix<T>::SquareMatrix(const size_t size) : 
        std::vector<std::vector<T>>(size, rowT(size))
    {
        // Nothing to do
    }

    template<class T>
    bool SquareMatrix<T>::saveToFile(const std::string & path) const
    {
        return saveToFile(std::ofstream(path));
    }

    template<class T>
    bool SquareMatrix<T>::saveToFile(std::ofstream & stream) const
    {
        auto retVal = stream.is_open();

        if (retVal == true)
        {
            std::ostream_iterator<boardFieldT> field_it(stream, " ");
            std::string space = " ";

            // Whole board
            for (size_t rowIdx = 0; rowIdx < size(); rowIdx++)
            {
                // Board fields
                std::copy((*this)[rowIdx].begin(), (*this)[rowIdx].end(), field_it);
                stream << std::endl;
            }
        }

        return retVal;
    }

    template<class T>
    bool SquareMatrix<T>::readFromFile(const std::string & path)
    {
        return readFromFile(std::ifstream(path));
    }

    template<class T>
    bool SquareMatrix<T>::readFromFile(std::ifstream & stream)
    {
        auto retVal = stream.is_open();

        if (retVal == true)
        {
            clear();
            std::string line;
            std::getline(stream, line);

            // Read all lines
            while (stream.good())
            {
                std::stringstream lineStream;
                lineStream << line;
                std::vector<T> row;
                T token;
                lineStream >> token;

                // Read all data from line
                while (lineStream.good())
                {
                    row.emplace_back(token);
                    lineStream >> token;
                }
                emplace_back(row);
                std::getline(stream, line);
            }

            size_t maxSize = size();
            for (const auto& row : *this)
            {
                if (maxSize < row.size())
                {
                    maxSize = row.size();
                }
            }

            // Ensure that it's square matrix
            resize(maxSize);
            for (auto& row : *this)
            {
                row.resize(maxSize);
            }
        }

        return retVal;
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
        column.reserve(size());

        for (auto& row : *this)
        {
            column.push_back(row[index]);
        }

        return column;
    }

    template<class T>
    inline const T SquareMatrix<T>::getCell(size_t row, size_t column) const
    {
        return (*this)[row][column];
    }

    template<class T>
    inline T SquareMatrix<T>::getCell(size_t row, size_t column)
    {
        return (*this)[row][column];
    }

    template<class T>
    inline void SquareMatrix<T>::setCell(size_t row, size_t column, T value)
    {
        (*this)[row][column] = value;
    }

    template<class T>
    inline SideE SquareMatrix<T>::whichEdgeRow(size_t row) const
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
    inline SideE SquareMatrix<T>::whichEdgeColumn(size_t column) const
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
    inline void SquareMatrix<T>::forEachCell(std::function<void(size_t, size_t)> function)
    {
        for (size_t row = 0; row < size(); row++)
        {
            for (size_t column = 0; column < size(); column++)
            {
                function(row, column);
            }
        }
    }

    template<class T>
    inline void SquareMatrix<T>::forEachVector(std::function<void(size_t, size_t)> function)
    {
        for (size_t rowAndColumn = 0; rowAndColumn < size(); rowAndColumn++)
        {
            function(rowAndColumn, rowAndColumn);
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
};