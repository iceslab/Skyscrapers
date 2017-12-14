#include <iostream>
#include "../Utilities/asserts.h"
#include "../Utilities/Timer.h"
#include "Combinatorics.h"
#include "Board.h"
#include "EfficientIncidenceCube.h"
#include "Solver.h"
#include "CpuSolver.h"
#include "ParallelCpuSolver.h"

//#define LOAD_FROM_FILE

using board::Board;
using solver::Solver;
using solver::CpuSolver;
using solver::ParallelCpuSolver;

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
    b.saveToFile("lastRun.txt");

    std::cout << "Expected result" << std::endl;
    b.print();
    std::cout << "==========================" << std::endl;

    std::cout << "Is board a valid solution?: " << b.checkValidityWithHints() << std::endl;

    CpuSolver c(b);
    ParallelCpuSolver pc(b);

    Timer time;
    time.start();
    const auto pcResult = pc.solve();
    auto milliseconds = time.stop(Resolution::MILLISECONDS);
    //std::cout << "Is Latin square?: " << pc.checkIfLatinSquare() << std::endl;
    //std::cout << "Is result board a valid solution?: " << pc.checkValidityWithHints() << std::endl;
    std::cout << "ParallelCpuSolver solving time: " << milliseconds << " ms" << std::endl;

    time.start();
    const auto cResult = c.solve();
    milliseconds = time.stop(Resolution::MILLISECONDS);

    std::cout << "Is Latin square?: " << c.checkIfLatinSquare() << std::endl;
    std::cout << "Is result board a valid solution?: " << c.checkValidityWithHints() << std::endl;
    std::cout << "CpuSolver solving time: " << milliseconds << " ms" << std::endl;

    std::cout << "ParallelCpuSolver results: " << std::endl;
    Solver::printResults(pcResult);

    std::cout << "CpuSolver results: "<< std::endl;
    Solver::printResults(cResult);

    //c.print();

    system("pause");
    return 0;
}