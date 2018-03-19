#pragma once
#include "../Utilities/asserts.h"
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <iostream>
#include <functional>
#include "SquareMatrix.h"
#include "EfficientIncidenceCube.h"

#define ENABLE_MEMOIZATION

namespace board
{
    // Typedefs for easier typing
    typedef unsigned int boardFieldT;
    typedef std::vector<std::vector<boardFieldT>> boardT;
    typedef std::vector<boardFieldT> hintT;
    typedef std::vector<boardFieldT> rowT;
    typedef std::set<boardFieldT> rowSetT;
    typedef std::vector<std::reference_wrapper<const boardFieldT>> columnConstT;
    typedef std::vector<std::reference_wrapper<boardFieldT>> columnT;
    typedef std::set<boardFieldT> columnSetT;
    typedef std::vector<boardFieldT> setIntersectionT;
    typedef std::vector<std::vector<bool>> memoizedSetValuesT;


    class Board : public matrix::SquareMatrix<boardFieldT>
    {
    public:
        static constexpr size_t hintSize = 4;
        std::array<hintT, hintSize> hints;

        Board(const Board & board);
        Board(Board && board);

        Board(const std::vector<boardFieldT> & fieldVector);

        Board(const size_t boardSize);
        Board(const std::string & path);
        Board(std::ifstream & stream);
        ~Board() = default;

        bool saveToFile(const std::string & path) const;
        bool saveToFile(std::ofstream & stream) const;

        bool readFromFile(const std::string & path);
        bool readFromFile(std::ifstream & stream);

        /// Generators

        // Generates latin square board 
        void generate();
        // Resizes and generates latin square board 
        void generate(const size_t boardSize);
        // Calculates hints
        void calculateHints();

        /// Operators
        bool operator==(const Board &other) const;
        bool operator!=(const Board &other) const;

        Board & operator=(const Board & board) = default;
        Board & operator=(Board && board) = default;

        /// Validators

        // Checks if board is latin square
        bool checkIfLatinSquare() const;
        // Checks validity of board in terms of hints 
        bool checkValidityWithHints() const;

        /// Hints manipulators
        // Gets visible buildings from given side and for given row or column
        boardFieldT getVisibleBuildings(matrix::SideE side, size_t rowOrColumn) const;
        // Gets visible buildings from given side and for given row or column assuming value at index
        boardFieldT getVisibleBuildingsIf(matrix::SideE side, size_t rowOrColumn, boardFieldT value, size_t index) const;
        // Returns if building can be placed in cell in terms of already placed buildings
        bool isBuildingPlaceable(size_t row, size_t column, boardFieldT building);
        // Returns if board is valid up tu given row and column in terms of already placed buildings and hints
        bool isBoardPartiallyValid(size_t row, size_t column);
        // Returns index of building in row which height == getSize(), if there is none returns getSize()
        boardFieldT locateHighestInRow(size_t row) const;
        // Returns index of building in column which height == getSize(), if there is none returns getSize()
        boardFieldT locateHighestInColumn(size_t column) const;

        /// Accessors

        // Sets cell to specific value only when it doesn't violate Latin square rules
        void setCell(size_t row, size_t column, boardFieldT value);
        void clearCell(size_t row, size_t column);

        boardFieldT getCell(size_t row, size_t column) const;

        size_t getSize() const;
        size_t getCellsCount() const;
        const memoizedSetValuesT & getSetRows() const;
        const memoizedSetValuesT & getSetColumns() const;
        void fill(const boardFieldT & value);

        matrix::SideE whichEdgeRow(size_t row) const;
        matrix::SideE whichEdgeColumn(size_t column) const;

        /// Output
        void print() const;
        void printToFile() const;
    private:
        static const std::array<matrix::SideE, 4> validSides;
        // Contains which values are set in each row and column
        memoizedSetValuesT setRows;
        memoizedSetValuesT setColumns;

        void resize(const size_t boardSize);

        static size_t allInstances;
        size_t instanceIndex;
    };

    // Counts visible buildings from "first" side
    template<class iterator_type>
    boardFieldT countVisibility(iterator_type first, iterator_type last)
    {
        size_t size = std::abs(first - last);
        size_t retVal = 1;
        size_t currentMax = 0;
        for (; first != last; first++)
        {
            if (*first == size)
                break;

            if (currentMax < *first)
            {
                currentMax = *first;
                retVal++;
            }
        }

        return static_cast<boardFieldT>(retVal);
    }
}