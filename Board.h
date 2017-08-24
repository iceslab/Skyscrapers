#pragma once
#include "macros.h"
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

    // Enum for accessing hints array
    enum HintsSideE
    {
        TOP = 0,
        RIGHT,
        BOTTOM,
        LEFT
    };

    const std::array<HintsSideE, 4> hintsArray;

    class Board : public matrix::SquareMatrix<boardFieldT>
    {
    public:
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

        /// Output
        void print() const;
    private:
        static constexpr size_t hintSize = 4;
        static const std::array<HintsSideE, 4> hintsArray;
        //boardT board;
        std::array<hintT, hintSize> hints;

        void resize(const boardFieldT boardSize);
        void fillWithZeros();

        /// Hints manipulators

        // <summary>as</summary>
        boardFieldT getVisibleBuildings(HintsSideE side, size_t rowOrColumn) const;
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