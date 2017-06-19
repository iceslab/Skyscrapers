#pragma once
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <random>

// Typedefs for easier typing
typedef uint32_t boardFieldT;
typedef std::vector<std::vector<boardFieldT>> boardT;
typedef std::vector<boardFieldT> hintT;
typedef std::vector<boardFieldT> rowT;
typedef std::set<boardFieldT> rowSetT;
typedef std::vector<std::reference_wrapper<boardFieldT>> columnT;
typedef std::set<boardFieldT> columnSetT;
typedef std::set<boardFieldT> differenceSetT;

// Enum for accessing hints array
enum HintsSide
{
	TOP = 0,
	RIGHT,
	BOTTOM,
	LEFT
};

class Board
{
public:
	Board(const boardFieldT boardSize);
	~Board() = default;

	void generate();
	void generate(const boardFieldT boardSize);

	// Operators
	bool operator==(const Board &other) const;
	bool operator!=(const Board &other) const;

	// Accessors
	size_t getSize() const;

	const rowT& getRow(size_t index) const;
	rowT& getRow(size_t index);

	columnT getColumn(size_t index);
private:
	static constexpr size_t hintSize = 4;
	boardT board;
	std::array<hintT, hintSize> hints;

	void resize(const boardFieldT boardSize);
	void fillWithZeros();
};

