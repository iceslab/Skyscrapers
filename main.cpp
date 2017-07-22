#include <iostream>
#include "macros.h"
#include "Combinatorics.h"
#include "Board.h"
#include "EfficientIncidenceCube.h"

int main(int argc, const char** argv)
{
    /*std::vector<int> A;
    for (int i = 1; i <= 3; i++)
    {
        A.push_back(i);
    }*/

    //Combinatorics::generateAllPermutations(A);

    board::Board b(4);
    b.generate();
    b.print();

    std::cout << "Is Latin square?: " << b.checkIfLatinSquare() << std::endl;
    std::cout << "Is valid solution?: " << b.checkValidityWithHints() << std::endl;

    system("pause");
    return 0;
}