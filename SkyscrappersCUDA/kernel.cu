#include "CUDAUtilities.cuh"
#include "asserts.h"
#include "Timer.h"
#include "../SequentialSolver.h"
#include "../ParallelCpuSolver.h"
#include "ParallelSolver.cuh"
#include <stdio.h>

CUDA_GLOBAL void parallelBoardSolving(cuda::solver::kernelInputT d_solvers,
                                      cuda::solver::kernelOutputT d_outputBoards,
                                      cuda::solver::kernelOutputSizesT d_outputBoardsSizes,
                                      int* lock)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    //printf("%s: Thread idx: %llu - begin\n", __FUNCTION__, idx);

    // Acquire lock
    //while (atomicCAS(lock, 1, 0) == 0);
    //for (size_t i = 0; i < blockDim.x; i++)
    //{
    //    __syncthreads();
    //    if (i == idx)
    //    {
    //        d_solvers[idx].getBoard().print(idx);
    //    }
    //}

    // Release lock
    //atomicExch(lock, 1);
    //if (idx == 2)
    {
        d_outputBoardsSizes[idx] =
            d_solvers[idx].solve(d_outputBoards + idx * cuda::solver::maxResultsPerThread, idx);
    }
    //printf("%s: Thread idx: %llu - finish\n", __FUNCTION__, idx);
}

//#define LOAD_FROM_FILE
#define BOARD_DIMENSIONS 6
#define STOP_LEVEL 4

int main(int argc, const char** argv)
{
    UNREFERENCED_PARAMETER(argc);
    UNREFERENCED_PARAMETER(argv);

    // Initialize device
    cuda::initDevice();

    // Prepare data on host
#ifndef LOAD_FROM_FILE
    board::Board b(BOARD_DIMENSIONS);
    b.generate();
#else
    board::Board b("input2.txt");
    b.calculateHints();
#endif // !LOAD_FROM_FILE
    b.saveToFile("lastRun.txt");

    printf("Expected result\n");
    b.print();
    printf("==========================\n");

    fflush(stdout);
    fflush(stderr);

    // CPU solvers
    solver::SequentialSolver c(b);
    solver::ParallelCpuSolver pc(b);

    Timer time;
    time.start();
    const auto pcResult = pc.solve(STOP_LEVEL);
    const auto pcMilliseconds = time.stop(Resolution::MILLISECONDS);

    time.start();
    const auto cResult = c.solve();
    const auto cMilliseconds = time.stop(Resolution::MILLISECONDS);

    
    // CPU solvers end

    // GPU solvers
    time.start();
    solver::ParallelSolver ps(b);
    printf("Generating boards...\n");
    const auto boards = ps.generateBoards(STOP_LEVEL);
    printf("Boards generated\n");
    size_t generatedSolversCount = 0;

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

    auto kernelLaunchMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto kernelSyncMilliseconds = std::numeric_limits<double>::quiet_NaN();

    // If allocation was successfull launch kernel
    if (cuda::solver::verifyAllocation(d_solvers, d_outputBoards, d_outputBoardsSizes))
    {
        dim3 numBlocks(1);
        dim3 threadsPerBlock(generatedSolversCount);
        unsigned int sharedMemorySize = 8 * sizeof(unsigned int);

        printf("Launching kernel...\n");
        fflush(stdout);
        fflush(stderr);

        int h_lock = 1;
        int* d_lock = nullptr;
        cudaError_t err = cudaMalloc(&d_lock, sizeof(int));
        if (err != cudaSuccess)
        {
            CUDA_PRINT_ERROR("Failed allocation", err);
            d_lock = nullptr;
        }
        else
        {
            err = cudaMemcpy(d_lock, &h_lock, sizeof(int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy", err);
                cudaFree(d_lock);
                d_lock = nullptr;
            }
        }

        Timer kernelTimer;
        kernelTimer.start();
        parallelBoardSolving << <numBlocks, threadsPerBlock, sharedMemorySize >> >
            (d_solvers,
             d_outputBoards,
             d_outputBoardsSizes,
             d_lock);

        kernelLaunchMilliseconds = kernelTimer.stop(Resolution::MILLISECONDS);
        cudaFree(d_lock);
        d_lock = nullptr;

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

    // Dellocating host memory (in reverse order)
    cuda::solver::freeHostResultArraySizes(h_outputBoardsSizes);
    cuda::solver::freeHostResultArray(h_outputBoards);

    // Dellocating device memory (in reverse order)
    cuda::solver::freeResultArraySizes(d_outputBoardsSizes);
    cuda::solver::freeResultArray(d_outputBoards);
    cuda::solver::freeSolvers(d_solvers);

    // Deinitialize device
    cuda::deinitDevice();
    const auto pgMilliseconds = time.stop(Resolution::MILLISECONDS);
    // GPU solvers end

    fflush(stdout);
    fflush(stderr);

    std::cout << "SequentialSolver solving time: " << cMilliseconds << " ms" << std::endl;
    std::cout << "ParallelCpuSolver solving time: " << pcMilliseconds << " ms" << std::endl;
    std::cout << "ParallelGpuSolver solving time: " << pgMilliseconds << " ms" << std::endl;
    std::cout << "Kernel launch time: " << kernelLaunchMilliseconds << " ms" << std::endl;
    std::cout << "Kernel synchronize time: " << kernelSyncMilliseconds << " ms" << std::endl;
    
    system("pause");
    return 0;
}

