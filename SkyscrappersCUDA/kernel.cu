#include "asserts.h"
#include "Timer.h"
#include "../Skyscrappers/SequentialSolver.h"
#include "../Skyscrappers/ParallelCpuSolver.h"
#include <stdio.h>
#include "XGetopt.h"
#include "Statistics.h"

#include "KernelFunctions.inl"

Statistics launchSequentialSolver(const board::Board & board);
Statistics launchParallelCpuSolver(const board::Board & board);
Statistics launchBaseParallelGpuSolver(const board::Board & board);

int main(int argc, char** argv)
{
    if (ProcessCommandLine(argc, argv) == FALSE)
    {
        // Exit when commandline processing fails
        printUsage();
        return -1;
    }

    printLaunchParameters();
    // Prepare data on host
    board::Board b(1);
    if (loadFromFile == true)
    {
        b.readFromFile(filePath);
        b.calculateHints();
    }
    else
    {
        b.generate(boardDimension);
    }
    b.saveToFile("lastRun.txt");

    printf("\nExpected result:\n");
    printf("==========================\n");
    b.print();
    printf("==========================\n");
    fflush(stdout);
    fflush(stderr);

    // CPU solvers
    const auto sStats = launchSequentialSolver(b);
    const auto pcStats = launchParallelCpuSolver(b);
    const auto bpgStats = launchBaseParallelGpuSolver(b);

    sStats.print();
    pcStats.print();
    bpgStats.print();

    //system("pause");
    return 0;
}

Statistics launchSequentialSolver(const board::Board & board)
{
    Statistics retVal;
    auto cMilliseconds = std::numeric_limits<double>::quiet_NaN();
    if (sequentialSolver == true)
    {
        solver::SequentialSolver c(board);
        Timer time;
        time.start();
        const auto cResult = c.solve();
        cMilliseconds = time.stop(Resolution::MILLISECONDS);
    }
    retVal.emplace_back("SequentialSolver solving time: ", cMilliseconds);
    return retVal;
}

Statistics launchParallelCpuSolver(const board::Board & board)
{
    Statistics retVal;
    auto pcMilliseconds = std::numeric_limits<double>::quiet_NaN();
    if (parallelCpuSolver == true)
    {
        solver::ParallelCpuSolver pc(board);
        Timer time;
        time.start();
        const auto pcResult = pc.solve(desiredBoards);
        pcMilliseconds = time.stop(Resolution::MILLISECONDS);
    }
    retVal.emplace_back("ParallelCpuSolver solving time: ", pcMilliseconds);
    return retVal;
}

Statistics launchBaseParallelGpuSolver(const board::Board & board)
{
    Statistics retVal;
    auto pgMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto initMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto deinitMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto generationMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto allocationMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto deallocationMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto kernelLaunchMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto kernelSyncMilliseconds = std::numeric_limits<double>::quiet_NaN();

    if (parallelGpuSolver == true)
    {
        Timer time;
        Timer timeInit;
        Timer timeGeneration;
        Timer timeAllocation;
        time.start();
        // Initialize device
        cuda::initDevice();
        initMilliseconds = time.stop(Resolution::MILLISECONDS);

        solver::ParallelSolver ps(board);
        timeGeneration.start();
        const auto boards = ps.generateNBoards(desiredBoards);
        generationMilliseconds = timeGeneration.stop(Resolution::MILLISECONDS);
        size_t generatedSolversCount = 0;

        timeAllocation.start();
        // Host vector for solvers - needed to properly execute destructors
        // It's lifetime ensures that pointers on device are valid during kernel execution
        std::vector<cuda::solver::SequentialSolver> h_solvers;
        // Host vector for boards - needed to properly execute destructors
        // It's lifetime ensures that pointers on device are valid during kernel execution
        std::vector<cuda::Board> h_boards;

        // Allocating memory on device
        auto d_solvers = cuda::solver::prepareSolvers(boards, h_solvers, generatedSolversCount);
        auto d_outputBoards = cuda::solver::prepareResultArray(h_boards, generatedSolversCount, boards.front().size());
        auto d_outputBoardsSizes = cuda::solver::prepareResultArraySizes(generatedSolversCount);

        // Allocating memory on host
        auto h_outputBoards = cuda::solver::prepareHostResultArray(generatedSolversCount);
        auto h_outputBoardsSizes = cuda::solver::prepareHostResultArraySizes(generatedSolversCount);
        allocationMilliseconds = timeAllocation.stop(Resolution::MILLISECONDS);


        // If allocation was successfull launch kernel
        if (cuda::solver::verifyAllocation(d_solvers, d_outputBoards, d_outputBoardsSizes))
        {
            dim3 numBlocks(1);
            dim3 threadsPerBlock(generatedSolversCount);

            printf("Launching kernel...\n");
            fflush(stdout);
            fflush(stderr);

            Timer kernelTimer;
            kernelTimer.start();
            parallelSolvingBase << <numBlocks, threadsPerBlock >> >
                (d_solvers,
                 d_outputBoards,
                 d_outputBoardsSizes);

            kernelLaunchMilliseconds = kernelTimer.stop(Resolution::MILLISECONDS);

            // Check for any errors launching the kernel
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                printf("parallelSolvingBase launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }
            else
            {
                // cudaDeviceSynchronize waits for the kernel to finish, and returns
                // any errors encountered during the launch.
                cudaStatus = cudaDeviceSynchronize();
                kernelSyncMilliseconds = kernelTimer.stop(Resolution::MILLISECONDS);
                fflush(stdout);
                fflush(stderr);
                if (cudaStatus != cudaSuccess)
                {
                    printf("cudaDeviceSynchronize returned %d \"%s\"\n",
                           cudaStatus,
                           cudaGetErrorString(cudaStatus));
                }
                //else
                //{
                //    cuda::solver::copyResultsArray(h_outputBoards,
                //                                   d_outputBoards,
                //                                   generatedSolversCount);
                //    cuda::solver::copyResultsArraySizes(h_outputBoardsSizes,
                //                                        d_outputBoardsSizes,
                //                                        generatedSolversCount);

                //    for (size_t i = 0; i < generatedSolversCount; i++)
                //    {
                //        const auto boardCount = h_outputBoardsSizes[i];
                //        DEBUG_PRINTLN("Result boards in thread %zu: %zu - max: %zu",
                //                      i,
                //                      boardCount,
                //                      cuda::solver::maxResultsPerThread);
                //        for (size_t j = 0; j < boardCount && j < cuda::solver::maxResultsPerThread; j++)
                //        {
                //            board::Board b(h_outputBoards[i * cuda::solver::maxResultsPerThread + j].getHostVector());
                //            b.calculateHints();
                //            b.print();
                //        }
                //    }
                //}
            }
        }


        timeAllocation.start();
        // Dellocating host memory (in reverse order)
        cuda::solver::freeHostResultArraySizes(h_outputBoardsSizes);
        cuda::solver::freeHostResultArray(h_outputBoards);

        // Dellocating device memory (in reverse order)
        cuda::solver::freeResultArraySizes(d_outputBoardsSizes);
        cuda::solver::freeResultArray(d_outputBoards);
        cuda::solver::freeSolvers(d_solvers);
        deallocationMilliseconds = timeAllocation.stop(Resolution::MILLISECONDS);

        timeInit.start();
        // Deinitialize device
        cuda::deinitDevice();
        deinitMilliseconds = timeInit.stop(Resolution::MILLISECONDS);
        pgMilliseconds = time.stop(Resolution::MILLISECONDS);
    }

    retVal.emplace_back("ParallelGpuSolver solving time: ", pgMilliseconds);
    retVal.emplace_back("\nDevice initialize time: ", initMilliseconds);
    retVal.emplace_back("Device deinitialize time: ", deinitMilliseconds);
    retVal.emplace_back("\nBoard generation time: ", generationMilliseconds);
    retVal.emplace_back("\nMemory allocation time: ", allocationMilliseconds);
    retVal.emplace_back("Memory deallocation time: ", deallocationMilliseconds);
    retVal.emplace_back("\nKernel launch time: ", kernelLaunchMilliseconds);
    retVal.emplace_back("Kernel synchronize time: ", kernelSyncMilliseconds);
    retVal.emplace_back("Allocation + synchronize + deallocation time: ",
                        allocationMilliseconds + kernelSyncMilliseconds + deallocationMilliseconds);
    return retVal;
}
