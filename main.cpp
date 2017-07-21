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

    EfficientIncidenceCube eic(3);
    eic.shuffle();
    std::cout << eic.toString() << std::endl;

    //board::Board b(4);
    //b.generate();
    //b.print();

    //system("pause");
    return 0;
}