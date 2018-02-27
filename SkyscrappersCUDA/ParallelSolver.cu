#include "ParallelSolver.cuh"

namespace cuda
{
    namespace solver
    {
        CUDA_HOST kernelInputT prepareSolvers(const std::vector<board::Board> & boards,
                                              std::vector<SequentialSolver> & h_solvers,
                                              size_t & count)
        {
            // Create array on host
            h_solvers.reserve(boards.size());

            for (auto & el : boards)
            {
                h_solvers.push_back(el);
            }

            // Create array on device
            kernelInputT d_retVal = nullptr;
            count = boards.size();
            cudaError_t err = cudaMalloc(&d_retVal, count * sizeof(SequentialSolver));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
                count = 0;
            }
            else
            {
                // Copy host array to device
                err = cudaMemcpy(d_retVal, h_solvers.data(), count * sizeof(SequentialSolver), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memcpy", err);
                    cudaFree(d_retVal);
                    d_retVal = nullptr;
                    count = 0;
                }
            }

            return d_retVal;
        }

        CUDA_HOST kernelOutputT prepareResultArray(std::vector<cuda::Board> & h_boards,
                                                   size_t solversCount,
                                                   size_t boardSize)
        {
            // Create array on host
            h_boards = std::vector<cuda::Board>(solversCount * maxResultsPerThread,
                                                cuda::Board(boardSize));

            kernelOutputT d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, solversCount * maxResultsPerThread * sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }
            else
            {
                // Zero out allocated memory
                err = cudaMemcpy(d_retVal,
                                 h_boards.data(),
                                 solversCount * maxResultsPerThread * sizeof(*d_retVal),
                                 cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memcpy", err);
                }
            }

            return d_retVal;
        }

        CUDA_HOST kernelOutputSizesT prepareResultArraySizes(size_t solversCount)
        {
            kernelOutputSizesT d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, solversCount * sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }
            else
            {
                // Zero out allocated memory
                err = cudaMemset(d_retVal, 0, solversCount * sizeof(*d_retVal));
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memset", err);
                }
            }

            return d_retVal;
        }

        CUDA_HOST kernelOutputT prepareHostResultArray(size_t solversCount)
        {
            kernelOutputT h_retVal = reinterpret_cast<kernelOutputT>(
                calloc(solversCount * maxResultsPerThread, sizeof(*h_retVal)));
            if (h_retVal == nullptr)
            {
                HOST_PRINT_ERROR("Failed calloc");
            }

            return h_retVal;
        }

        CUDA_HOST kernelOutputSizesT prepareHostResultArraySizes(size_t solversCount)
        {
            kernelOutputSizesT h_retVal = reinterpret_cast<kernelOutputSizesT>(
                calloc(solversCount, sizeof(*h_retVal)));
            if (h_retVal == nullptr)
            {
                HOST_PRINT_ERROR("Failed calloc");
            }

            return h_retVal;
        }

        CUDA_HOST void freeSolvers(kernelInputT & d_solvers)
        {
            cudaFree(d_solvers);
            d_solvers = nullptr;
        }

        CUDA_HOST void freeResultArray(kernelOutputT & d_outputBoards)
        {
            cudaFree(d_outputBoards);
            d_outputBoards = nullptr;
        }

        CUDA_HOST void freeResultArraySizes(kernelOutputSizesT & d_outputBoardsSizes)
        {
            cudaFree(d_outputBoardsSizes);
            d_outputBoardsSizes = nullptr;
        }

        CUDA_HOST void freeHostResultArray(kernelOutputT & h_outputBoards)
        {
            free(h_outputBoards);
            h_outputBoards = nullptr;
        }

        CUDA_HOST void freeHostResultArraySizes(kernelOutputSizesT & h_outputBoardsSizes)
        {
            free(h_outputBoardsSizes);
            h_outputBoardsSizes = nullptr;
        }

        CUDA_HOST void copyResultsArray(kernelOutputT h_outputBoards,
                                        kernelOutputT d_outputBoards,
                                        size_t solversCount)
        {
            cudaError_t err = cudaMemcpy(h_outputBoards,
                                         d_outputBoards,
                                         solversCount * maxResultsPerThread * sizeof(*h_outputBoards),
                                         cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy", err);
            }
        }

        CUDA_HOST void copyResultsArraySizes(kernelOutputSizesT h_outputBoardsSizes,
                                             kernelOutputSizesT d_outputBoardsSizes,
                                             size_t solversCount)
        {
            cudaError_t err = cudaMemcpy(h_outputBoardsSizes,
                                         d_outputBoardsSizes,
                                         solversCount * sizeof(*h_outputBoardsSizes),
                                         cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy", err);
            }
        }

        CUDA_HOST bool verifyAllocation(kernelInputT & d_solvers,
                                        kernelOutputT & d_outputBoards,
                                        kernelOutputSizesT & d_outputBoardsSizes)
        {
            return d_solvers != nullptr &&
                d_outputBoards != nullptr &&
                d_outputBoardsSizes != nullptr;
        }

        //CUDA_GLOBAL void parallelBoardSolving(kernelInputT d_solvers,
        //                                      kernelOutputT d_outputBoards,
        //                                      kernelOutputSizesT d_outputBoardsSizes,
        //                                      stackT d_stack)
        //{
        //    // It denotes thread index and array index
        //    const auto idx = threadIdx.x;
        //    printf("Thread idx: %u\n");
        //    //d_outputBoardsSizes[idx] = d_solvers[idx].solve(d_outputBoards + idx * maxResultsPerThread, d_stack + idx);
        //}
    }
}
