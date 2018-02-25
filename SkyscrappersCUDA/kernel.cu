#include "CUDAUtilities.cuh"
#include "../ParallelSolver.h"
#include "ParallelSolver.cuh"
#include <stdio.h>

CUDA_GLOBAL void testKernel1()
{
    const auto idx = threadIdx.x;
    printf("%s: Thread idx: %u\n", __FUNCTION__);
}

CUDA_GLOBAL void testKernel2(cuda::solver::kernelInputT d_solvers,
                             cuda::solver::kernelOutputT d_outputBoards,
                             cuda::solver::kernelOutputSizesT d_outputBoardsSizes,
                             cuda::solver::stackT d_stack)
{
    const auto idx = threadIdx.x;
    printf("%s: Thread idx: %u\n", __FUNCTION__);
}

CUDA_GLOBAL void parallelBoardSolving(cuda::solver::kernelInputT d_solvers,
                                      cuda::solver::kernelOutputT d_outputBoards,
                                      cuda::solver::kernelOutputSizesT d_outputBoardsSizes,
                                      cuda::solver::stackPtrT d_stack)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    printf("%s: Thread idx: %u - begin\n", __FUNCTION__);
    d_outputBoardsSizes[idx] = d_solvers[idx].solve(d_outputBoards + idx * cuda::solver::maxResultsPerThread, d_stack + idx);
    printf("%s: Thread idx: %u - finish\n", __FUNCTION__);
}

#define LOAD_FROM_FILE

int main(int argc, const char** argv)
{
    UNREFERENCED_PARAMETER(argc);
    UNREFERENCED_PARAMETER(argv);

    // Initialize device
    cuda::initDevice();

    // Prepare data on host
#ifndef LOAD_FROM_FILE
    board::Board b(6);
    b.generate();
#else
    board::Board b("input.txt");
    b.calculateHints();
#endif // !LOAD_FROM_FILE
    b.saveToFile("lastRun.txt");

    solver::ParallelSolver ps(b);
    const auto boards = ps.generateBoards(1);
    size_t generatedSolversCount = 0;

    // Host vector for solvers - needed to properly execute destructors
    // It's lifetime ensures that pointers on device are valid during kernel execution
    std::vector<cuda::solver::SequentialSolver> h_solvers;

    // Allocating memory on device
    auto d_solvers = cuda::solver::prepareSolvers(boards, h_solvers, generatedSolversCount);
    auto d_outputBoards = cuda::solver::prepareResultArray(generatedSolversCount);
    auto d_outputBoardsSizes = cuda::solver::prepareResultArraySizes(generatedSolversCount);
    auto d_stack = cuda::solver::prepareStack(b.getSize(), generatedSolversCount);

    //auto d_solvers = cuda::solver::kernelInputT(nullptr);
    //auto d_outputBoards = cuda::solver::kernelOutputT(nullptr);
    //auto d_outputBoardsSizes = cuda::solver::kernelOutputSizesT(nullptr);
    //auto d_stack = cuda::solver::stackT(nullptr);

    // Allocating memory on host
    auto h_outputBoards = cuda::solver::prepareHostResultArray(generatedSolversCount);
    auto h_outputBoardsSizes = cuda::solver::prepareHostResultArraySizes(generatedSolversCount);

    // If allocation was successfull launch kernel
    if (cuda::solver::verifyAllocation(d_solvers, d_outputBoards, d_outputBoardsSizes, d_stack))
    {
        dim3 numBlocks(1);
        dim3 threadsPerBlock(generatedSolversCount);
        unsigned int sharedMemorySize = 8 * sizeof(unsigned int);
        
        //testKernel1<<<numBlocks, threadsPerBlock, sharedMemorySize>>>();
        /*testKernel2<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_solvers,
                                                                      d_outputBoards,
                                                                      d_outputBoardsSizes,
                                                                      d_stack);*/
        parallelBoardSolving<<<numBlocks, threadsPerBlock, sharedMemorySize>>>
            (d_solvers,
             d_outputBoards,
             d_outputBoardsSizes,
             d_stack);

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
            if (cudaStatus != cudaSuccess)
            {
                printf("cudaDeviceSynchronize returned %d \"%s\"\n",
                       cudaStatus,
                       cudaGetErrorString(cudaStatus));
            }
            else
            {
                cuda::solver::copyResultsArray(h_outputBoards,
                                               d_outputBoards,
                                               generatedSolversCount);
                cuda::solver::copyResultsArraySizes(h_outputBoardsSizes,
                                                    d_outputBoardsSizes,
                                                    generatedSolversCount);

                for (size_t i = 0; i < generatedSolversCount; i++)
                {
                    const auto boardCount = h_outputBoardsSizes[i];
                    DEBUG_PRINTLN("Result boards in thread %zu: %zu - max: %zu",
                                  i,
                                  boardCount,
                                  cuda::solver::maxResultsPerThread);
                    for (size_t j = 0; j < boardCount && j < cuda::solver::maxResultsPerThread; j++)
                    {
                        board::Board b(h_outputBoards[i * cuda::solver::maxResultsPerThread + j].getHostVector());
                        b.print();
                    }
                }
            }
        }
    }

    // Dellocating host memory (in reverse order)
    cuda::solver::freeHostResultArraySizes(h_outputBoardsSizes);
    cuda::solver::freeHostResultArray(h_outputBoards);

    // Dellocating device memory (in reverse order)
    cuda::solver::freeStack(d_stack);
    cuda::solver::freeResultArraySizes(d_outputBoardsSizes);
    cuda::solver::freeResultArray(d_outputBoards);
    cuda::solver::freeSolvers(d_solvers);

    // Deinitialize device
    cuda::deinitDevice();

    system("pause");
    return 0;
}

