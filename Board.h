#pragma once
#include <array>
#include <vector>

// Typedefs for easier typing
typedef uint32_t boardFieldT;
typedef std::vector<std::vector<boardFieldT>> boardT;
typedef std::vector<boardFieldT> hintT;
typedef hintT rowT;
typedef hintT columnT;

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
private:
	static constexpr size_t hintSize = 4;
	boardT board;
	std::array<hintT, hintSize> hints;

	void resize(const boardFieldT boardSize);
};

