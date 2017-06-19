#include <iostream>
#include "macros.h"
#include "Combinatorics.h"

int main(int argc, const char** argv)
{
	std::vector<int> A;
	for (int i = 1; i <= 3; i++)
	{
		A.push_back(i);
	}

	Combinatorics::generateAllPermutations(A);

	system("pause");
	return 0;
}