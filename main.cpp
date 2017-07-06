#include <iostream>
#include "macros.h"
#include "Combinatorics.h"
#include "Board.h"

int main(int argc, const char** argv)
{
	/*std::vector<int> A;
	for (int i = 1; i <= 3; i++)
	{
		A.push_back(i);
	}*/

	//Combinatorics::generateAllPermutations(A);

	board::Board b(3);
	b.generate();
	b.print();

	//system("pause");
	return 0;
}