#include "CUDAUtilities.cuh"
#include "asserts.h"
#include "Timer.h"
#include "../SequentialSolver.h"
#include "../ParallelCpuSolver.h"
#include "ParallelSolver.cuh"
#include <stdio.h>
#include "XGetopt.h"

CUDA_GLOBAL void parallelBoardSolving(cuda::solver::kernelInputT d_solvers,
                                      cuda::solver::kernelOutputT d_outputBoards,
                                      cuda::solver::kernelOutputSizesT d_outputBoardsSizes)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    d_outputBoardsSizes[idx] =
        d_solvers[idx].solve(d_outputBoards + idx * cuda::solver::maxResultsPerThread, idx);
}

int main(int argc, char** argv)
{
    ProcessCommandLine(argc, argv);

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

    printf("Expected result\n");
    b.print();
    printf("==========================\n");
    fflush(stdout);
    fflush(stderr);

    // CPU solvers
    auto cMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto pcMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto pgMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto initMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto deinitMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto generationMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto allocationMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto deallocationMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto kernelLaunchMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto kernelSyncMilliseconds = std::numeric_limits<double>::quiet_NaN();

    if (parallelCpuSolver == true)
    {
        solver::ParallelCpuSolver pc(b);
        Timer time;
        time.start();
        const auto pcResult = pc.solve(desiredBoards);
        pcMilliseconds = time.stop(Resolution::MILLISECONDS);
    }

    if (sequentialSolver == true)
    {
        solver::SequentialSolver c(b);
        Timer time;
        time.start();
        const auto cResult = c.solve();
        cMilliseconds = time.stop(Resolution::MILLISECONDS);
    }
    // CPU solvers end

    // GPU solver
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

        solver::ParallelSolver ps(b);
        printf("Generating boards...\n");

        timeGeneration.start();
        const auto boards = ps.generateNBoards(desiredBoards);
        generationMilliseconds = timeGeneration.stop(Resolution::MILLISECONDS);

        printf("Boards generated: %zu\n", boards.size());
        size_t generatedSolversCount = 0;

        timeAllocation.start();
        // Host vector for solvers - needed to properly execute destructors
        // It's lifetime ensures that pointers on device are valid during kernel execution
        std::vector<cuda::solver::SequentialSolver> h_solvers;
        // Host vector for boards - needed to properly execute destructors
        // It's lifetime ensures that pointers on device are valid during kernel execution
        std::vector<cuda::Board> h_boards;

        // Allocating memory on device
        //printf("Allocating memory on device...\n");
        auto d_solvers = cuda::solver::prepareSolvers(boards, h_solvers, generatedSolversCount);
        auto d_outputBoards = cuda::solver::prepareResultArray(h_boards, generatedSolversCount, boards.front().size());
        auto d_outputBoardsSizes = cuda::solver::prepareResultArraySizes(generatedSolversCount);
        //printf("Memory allocated\n");

        // Allocating memory on host
        //printf("Allocating memory on host...\n");
        auto h_outputBoards = cuda::solver::prepareHostResultArray(generatedSolversCount);
        auto h_outputBoardsSizes = cuda::solver::prepareHostResultArraySizes(generatedSolversCount);
        //printf("Memory allocated\n");
        allocationMilliseconds = timeAllocation.stop(Resolution::MILLISECONDS);


        // If allocation was successfull launch kernel
        if (cuda::solver::verifyAllocation(d_solvers, d_outputBoards, d_outputBoardsSizes))
        {
            dim3 numBlocks(1);
            dim3 threadsPerBlock(generatedSolversCount);
            unsigned int sharedMemorySize = 8 * sizeof(unsigned int);

            printf("Launching kernel...\n");
            fflush(stdout);
            fflush(stderr);

            Timer kernelTimer;
            kernelTimer.start();
            parallelBoardSolving << <numBlocks, threadsPerBlock, sharedMemorySize >> >
                (d_solvers,
                 d_outputBoards,
                 d_outputBoardsSizes);

            kernelLaunchMilliseconds = kernelTimer.stop(Resolution::MILLISECONDS);

            // Check for any errors launching the kernel
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                printf("parallelBoardSolving launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
    // GPU solvers end

    fflush(stdout);
    fflush(stderr);

    std::cout << "SequentialSolver solving time: " << cMilliseconds << " ms" << std::endl;
    std::cout << "ParallelCpuSolver solving time: " << pcMilliseconds << " ms" << std::endl;
    std::cout << "ParallelGpuSolver solving time: " << pgMilliseconds << " ms" << std::endl;

    std::cout << "\nDevice initialize time: " << initMilliseconds << " ms" << std::endl;
    std::cout << "Device deinitialize time: " << deinitMilliseconds << " ms" << std::endl;

    std::cout << "\nBoard generation time: " << generationMilliseconds << " ms" << std::endl;

    std::cout << "\nMemory allocation time: " << allocationMilliseconds << " ms" << std::endl;
    std::cout << "Memory deallocation time: " << deallocationMilliseconds << " ms" << std::endl;

    std::cout << "\nKernel launch time: " << kernelLaunchMilliseconds << " ms" << std::endl;
    std::cout << "Kernel synchronize time: " << kernelSyncMilliseconds << " ms" << std::endl;

    std::cout << "Allocation + synchronize + deallocation time: "
        << allocationMilliseconds + kernelSyncMilliseconds + deallocationMilliseconds
        << " ms" << std::endl;

    //system("pause");
    return 0;
}

