#include "Board.h"

Board::Board(const boardFieldT boardSize) : board(boardSize, rowT(boardSize))
{
	// Resize hints
	for (auto& h : hints)
	{
		h.resize(boardSize);
	}
}

void Board::generate()
{
	generate(board.size());
}

void Board::generate(const boardFieldT boardSize)
{
	resize(boardSize);
	fillWithZeros();

	// Fill first row with values from 1 to board size
	auto& firstRow = board.front();
	for (size_t i = 1; i <= getSize(); i++)
	{
		firstRow[i] = i;
	}

	// Randomize first row
	std::random_shuffle(firstRow.begin(), firstRow.end());

	// Prepare column sets for finding available values
	std::vector<columnSetT> columnSets;
	columnSets.reserve(getSize());

	for (size_t i = 0; i < getSize(); i++)
	{
		columnSets[i].insert(firstRow[i]);
	}

	// For each row...
	for (size_t rowIdx = 1; rowIdx <= getSize(); rowIdx++)
	{
		// ... and each column...
		rowSetT rowSet;
		for (size_t columnIdx = 0; columnIdx <= getSize(); columnIdx++)
		{
			auto& columnSet = columnSets[columnIdx];
			differenceT difference;

			// ... find which values are available for their intersection
			std::set_symmetric_difference(rowSet.begin(), rowSet.end(), columnSet.begin(), columnSet.end(), difference.begin());

			// Randomly choose one of the values
			// TODO

			// Update row and column sets and board itself
			// TODO
		}
	}
}

bool Board::operator==(const Board & other) const
{
	if (board != other.board)
	{
		return false;
	}

	for (size_t i = 0; i < hintSize; i++)
	{
		if (hints[i] != other.hints[i])
			return false;
	}
	
	return true;
}

bool Board::operator!=(const Board & other) const
{
	return !(*this == other);
}

size_t Board::getSize() const
{
	return board.size();
}

const rowT & Board::getRow(size_t index) const
{
	return board[index];
}

rowT & Board::getRow(size_t index)
{
	return board[index];
}

columnT Board::getColumn(size_t index) 
{
	columnT column;
	column.reserve(getSize());

	for (auto& row : board)
	{
		column.push_back(row[index]);
	}

	return column;
}

void Board::resize(const boardFieldT boardSize)
{
	if (boardSize == board.size())
		return;

	// Resize rows count
	board.resize(boardSize);
	for (auto& row : board)
	{
		// Resize rows
		row.resize(boardSize);
	}

	for (auto& h : hints)
	{
		// Resize hints
		h.resize(boardSize);
	}
}

void Board::fillWithZeros()
{
	for (auto& row : board)
	{
		std::fill(row.begin(), row.end(), boardFieldT());
	}
}


