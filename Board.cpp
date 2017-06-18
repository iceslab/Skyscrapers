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


