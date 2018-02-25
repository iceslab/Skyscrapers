#include <iostream>
#include "asserts.h"
#include "Timer.h"
#include "Combinatorics.h"
#include "Board.h"
#include "EfficientIncidenceCube.h"
#include "Solver.h"
#include "SequentialSolver.h"
#include "ParallelCpuSolver.h"
#include "ParallelGpuSolver.h"

#define LOAD_FROM_FILE

using board::Board;
using solver::Solver;
using solver::SequentialSolver;
using solver::ParallelCpuSolver;
using solver::ParallelGpuSolver;
using utils::AMPUtilities;

int main(int argc, const char** argv)
{
    UNREFERENCED_PARAMETER(argc);
    UNREFERENCED_PARAMETER(argv);

#ifndef LOAD_FROM_FILE
    Board b(6);
    b.generate();
#else
    Board b("input.txt");
    b.calculateHints();
#endif // !LOAD_FROM_FILE
    b.saveToFile("lastRun.txt");

    std::cout << "Expected result" << std::endl;
    b.print();
    std::cout << "==========================" << std::endl;

    std::cout << "Is board a valid solution?: " << AMPUtilities::boolToString(b.checkValidityWithHints()) << std::endl;

    SequentialSolver c(b);
    ParallelCpuSolver pc(b);

    Timer time;
    time.start();
    //const auto pcResult = pc.solve();
    auto milliseconds = time.stop(Resolution::MILLISECONDS);
    //std::cout << "Is Latin square?: " << pc.checkIfLatinSquare() << std::endl;
    //std::cout << "Is result board a valid solution?: " << pc.checkValidityWithHints() << std::endl;
    std::cout << "ParallelCpuSolver solving time: " << milliseconds << " ms" << std::endl;

    time.start();
    const auto cResult = c.solve();
    milliseconds = time.stop(Resolution::MILLISECONDS);

    //std::cout << "Is Latin square?: " << AMPUtilities::boolToString(c.checkIfLatinSquare()) << std::endl;
    //std::cout << "Is result board a valid solution?: " << AMPUtilities::boolToString(c.checkValidityWithHints()) << std::endl;
    std::cout << "SequentialSolver solving time: " << milliseconds << " ms" << std::endl;

    //std::cout << "\nParallelCpuSolver results: " << std::endl;
    //Solver::printResults(pcResult);

    //std::cout << "\nCpuSolver results: "<< std::endl;
    //Solver::printResults(cResult);

    //c.print();

    //const auto equalSizes = pcResult.size() == cResult.size();
    //std::cout << "Are results sizes equal?: " << AMPUtilities::boolToString(equalSizes) << std::endl;
    std::cout << "Sizes: " << std::endl;
    //std::cout << "ParallelCpuSolver: " << pcResult.size() << std::endl;
    //std::cout << "ParallelGpuSolver: " << pgResult.size() << std::endl;
    std::cout << "SequentialSolver: " << cResult.size() << std::endl;

    system("pause");
    return 0;
}
