
#include "CUDAUtilities.cuh"
#include "../ParallelSolver.h"
#include "ParallelSolver.cuh"
#include <stdio.h>

int main(int argc, const char** argv)
{
    UNREFERENCED_PARAMETER(argc);
    UNREFERENCED_PARAMETER(argv);

    // Initialize device
    cuda::initDevice();

    // Prepare data on host
    board::Board b(6);
    b.generate();

    solver::ParallelSolver ps(b);
    const auto boards = ps.generateBoards(4);
    size_t generatedSolversCount = 0;

    // Allocating memory
    auto d_solvers = cuda::solver::prepareSolvers(boards, generatedSolversCount);
    auto d_outputBoards = cuda::solver::prepareResultArray(generatedSolversCount);
    auto d_outputBoardsSizes = cuda::solver::prepareResultArraySizes(generatedSolversCount);
    auto d_stack = cuda::solver::prepareStack(b.getSize());

    // If allocation was successfull launch kernel
    if (cuda::solver::verifyAllocation(d_solvers, d_outputBoards, d_outputBoardsSizes, d_stack))
    {
        cuda::solver::parallelBoardSolving << <1, generatedSolversCount >> > (d_solvers,
                                                                              d_outputBoards,
                                                                              d_outputBoardsSizes,
                                                                              d_stack);
    }

    // Dellocating memory (in reverse order)
    cuda::solver::freeStack(d_stack);
    cuda::solver::freeResultArraySizes(d_outputBoardsSizes);
    cuda::solver::freeResultArray(d_outputBoards);
    cuda::solver::freeSolvers(d_solvers);

    // Deinitialize device
    cuda::deinitDevice();

    return 0;
}
