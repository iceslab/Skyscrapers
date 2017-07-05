#include "Board.h"

using namespace board;

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
	for (size_t i = 0; i < getSize(); i++)
	{
		firstRow[i] = i + 1;
	}

	// Randomize first row
	std::random_shuffle(firstRow.begin(), firstRow.end());

	// Prepare column sets for finding available values
	std::vector<columnSetT> columnSets(getSize());
	//columnSets.reserve();

	for (size_t i = 0; i < getSize(); i++)
	{
		columnSets[i].insert(firstRow[i]);
	}

	// For each row...
	for (size_t rowIdx = 1; rowIdx < getSize(); rowIdx++)
	{
		// This is full, because every iteration of loop, new row is chosen
		rowSetT rowSet(firstRow.begin(), firstRow.end());

		// ... and each column...
		for (size_t columnIdx = 0; columnIdx < getSize(); columnIdx++)
		{
			auto& columnSet = columnSets[columnIdx];
			differenceT difference;

			// ... find which values are available for their intersection
			std::set_difference(rowSet.begin(),
								rowSet.end(),
								columnSet.begin(),
								columnSet.end(),
								std::back_inserter(difference));

			// Randomly choose one of the values
			std::random_shuffle(difference.begin(), difference.end());
			auto& value = difference.front();

			// Update row and column sets and board itself
			rowSet.erase(value);
			columnSet.insert(value);
			board[rowIdx][columnIdx] = value;
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

bool Board::checkValidity() const
{
	return false;
}

bool Board::checkValidityWithHints() const
{
	return false;
}

void Board::print() const
{
	std::ostream_iterator<std::string> space_it(std::cout, " ");
	std::ostream_iterator<boardFieldT> field_it(std::cout, " ");
	std::string space = " ";

	// Free field to align columns
	std::cout << "  ";
	// Top hints
	std::copy(hints[TOP].begin(), hints[TOP].end(), field_it);
	std::cout << std::endl;

	// Whole board
	for (size_t rowIdx = 0; rowIdx < getSize(); rowIdx++)
	{
		// Left hint field
		std::copy(hints[LEFT].begin() + rowIdx, hints[LEFT].begin() + rowIdx + 1, field_it);
		
		// Board fields
		std::copy(board[rowIdx].begin(), board[rowIdx].end(), field_it);

		// Right hint field
		std::copy(hints[RIGHT].begin() + rowIdx, hints[RIGHT].begin() + rowIdx + 1, field_it);
		std::cout << std::endl;
	}

	// Free field to align columns
	std::cout << "  ";
	// Bottom hints
	std::copy(hints[BOTTOM].begin(), hints[BOTTOM].end(), field_it);
	std::cout << std::endl;
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

boardFieldT Board::getVisibleBuildings(HintsSide side, size_t rowOrColumn)
{
	boardFieldT retVal = 0;
	auto& row = getRow(rowOrColumn);
	auto& column = getColumn(rowOrColumn);
	switch (side)
	{
		case TOP:
			retVal = countVisibility(column.begin(), column.end());
			break;
		case RIGHT:
			retVal = countVisibility(row.rbegin(), row.rend());
			break;
		case BOTTOM:
			retVal = countVisibility(column.rbegin(), column.rend());
			break;
		case LEFT:
			retVal = countVisibility(row.begin(), row.end());
			break;
		default:
			
			break;
	}

	return retVal;
}


