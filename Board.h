#pragma once
#include "../Utilities/asserts.h"
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <iostream>
#include <iterator>
#include <functional>
#include "SquareMatrix.h"
#include "EfficientIncidenceCube.h"

namespace board
{
    // Typedefs for easier typing
    typedef uint32_t boardFieldT;
    typedef std::vector<std::vector<boardFieldT>> boardT;
    typedef std::vector<boardFieldT> hintT;
    typedef std::vector<boardFieldT> rowT;
    typedef std::set<boardFieldT> rowSetT;
    typedef std::vector<std::reference_wrapper<const boardFieldT>> columnConstT;
    typedef std::vector<std::reference_wrapper<boardFieldT>> columnT;
    typedef std::set<boardFieldT> columnSetT;
    typedef std::vector<boardFieldT> setIntersectionT;

    class Board : public matrix::SquareMatrix<boardFieldT>
    {
    public:
        static constexpr size_t hintSize = 4;
        std::array<hintT, hintSize> hints;

        Board(const boardFieldT boardSize);
        ~Board() = default;

        /// Generators

        // Generates latin square board 
        void generate();
        // Resizes and generates latin square board 
        void generate(const boardFieldT boardSize);

        /// Operators
        bool operator==(const Board &other) const;
        bool operator!=(const Board &other) const;

        /// Validators

        // Checks if board is latin square
        bool checkIfLatinSquare() const;
        // Checks validity of board in terms of hints 
        bool checkValidityWithHints() const;

        // Hints manipulators
        // Gets visible buildings from given side and for given row or column
        boardFieldT getVisibleBuildings(matrix::SideE side, size_t rowOrColumn) const;
        // Gets visible buildings from given side and for given row or column assuming value at index
        boardFieldT getVisibleBuildingsIf(matrix::SideE side, size_t rowOrColumn, boardFieldT value, size_t index) const;
        // Returns if building can be placed in cell in terms of already placed buildings
        bool isBuildingPlaceable(size_t row, size_t column, boardFieldT building);
        // Returns index of building in row which height == size(), if there is none returns size()
        boardFieldT locateHighestInRow(size_t row) const;
        // Returns index of building in column which height == size(), if there is none returns size()
        boardFieldT locateHighestInColumn(size_t column) const;

        // Output
        void print() const;
    private:
        static const std::array<matrix::SideE, 4> validSides;

        void resize(const boardFieldT boardSize);
    };

    // Counts visible buildings from "first" side
    template<class iterator_type>
    size_t countVisibility(iterator_type first, iterator_type last)
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

        return retVal;
    }
}