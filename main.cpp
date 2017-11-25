#include <iostream>
#include "../Utilities/asserts.h"
#include "../Utilities/Timer.h"
#include "Combinatorics.h"
#include "Board.h"
#include "EfficientIncidenceCube.h"
#include "Solver.h"
#include "CpuSolver.h"

//#define LOAD_FROM_FILE

using board::Board;
using solver::CpuSolver;

int main(int argc, const char** argv)
{
    UNREFERENCED_PARAMETER(argc);
    UNREFERENCED_PARAMETER(argv);

#ifndef LOAD_FROM_FILE
    Board b(7);
    b.generate();
#else
    Board b("input.txt");
    b.calculateHints();
#endif // !LOAD_FROM_FILE

    std::cout << "Expected result" << std::endl;
    b.print();
    std::cout << "==========================" << std::endl;

    std::cout << "Is board a valid solution?: " << b.checkValidityWithHints() << std::endl;

    CpuSolver c(b);

    Timer time;
    time.start();
    c.solve();
    const auto milliseconds = time.stop(Resolution::MILLISECONDS);
    std::cout << "Is Latin square?: " << c.checkIfLatinSquare() << std::endl;
    std::cout << "Is result board a valid solution?: " << c.checkValidityWithHints() << std::endl;
    std::cout << "Solving time: " << milliseconds << " ms" << std::endl;

    //c.print();

    system("pause");
    return 0;
}