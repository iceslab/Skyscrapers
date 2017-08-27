#include <iostream>
#include "macros.h"
#include "Combinatorics.h"
#include "Board.h"
#include "EfficientIncidenceCube.h"
#include "Solver.h"
#include "CpuSolver.h"

using board::Board;
using solver::CpuSolver;

int main(int argc, const char** argv)
{
    Board b(4);
    b.generate();
    b.print();

    std::cout << "Is Latin square?: " << b.checkIfLatinSquare() << std::endl;
    std::cout << "Is valid solution?: " << b.checkValidityWithHints() << std::endl;

    CpuSolver c(b);
    
    c.solve();
    c.print();

    system("pause");
    return 0;
}