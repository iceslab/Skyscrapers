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
            h_boards = std::vector<cuda::Board>(CUDA_MAX_RESULTS,
                                                cuda::Board(boardSize));

            kernelOutputT d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, CUDA_MAX_RESULTS * sizeof(*d_retVal));
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
                                 CUDA_MAX_RESULTS * sizeof(*d_retVal),
                                 cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memcpy", err);
                }
            }

            return d_retVal;
        }

        CUDA_HOST uint32T * prepareResultArraySize()
        {
            uint32T* d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }
            else
            {
                uint32T h_valueToSet = 1;
                // Copy host variable to device
                err = cudaMemcpy(d_retVal, &h_valueToSet, sizeof(*d_retVal), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memcpy", err);
                    cudaFree(d_retVal);
                    d_retVal = nullptr;
                }
            }
            return d_retVal;
        }

        CUDA_HOST threadLocalsT * prepareThreadLocals(size_t solversCount)
        {
            threadLocalsT* d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, solversCount * sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }
            else
            {
                // Copy host array to device
                err = cudaMemset(d_retVal, 0, solversCount * sizeof(*d_retVal));
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memset", err);
                    cudaFree(d_retVal);
                    d_retVal = nullptr;
                }
            }

            return d_retVal;
        }

        CUDA_HOST uint32T * prepareScatterArray(size_t solversCount)
        {
            uint32T* d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, solversCount * sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }

            return d_retVal;
        }

        CUDA_HOST cudaEventsDeviceT * prepareCudaEventDevice(const std::vector<cudaEventsDeviceT>& h_timers)
        {
            cudaEventsDeviceT* d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, h_timers.size() * sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }

            return d_retVal;
        }

        CUDA_HOST kernelOutputT prepareHostResultArray()
        {
            kernelOutputT h_retVal = reinterpret_cast<kernelOutputT>(
                calloc(CUDA_MAX_RESULTS, sizeof(*h_retVal)));
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

        CUDA_HOST void freeResultArraySize(uint32T* & d_outputBoardsSize)
        {
            cudaFree(d_outputBoardsSize);
            d_outputBoardsSize = nullptr;
        }

        CUDA_HOST void freeThreadLocals(threadLocalsT *& d_threadLocals)
        {
            cudaFree(d_threadLocals);
            d_threadLocals = nullptr;
        }

        CUDA_HOST void freeScatterArray(uint32T *& d_scatterArray)
        {
            cudaFree(d_scatterArray);
            d_scatterArray = nullptr;
        }

        CUDA_HOST void freeCudaEventDevice(cudaEventsDeviceT *& d_timers)
        {
            cudaFree(d_timers);
            d_timers = nullptr;
        }

        CUDA_HOST void freeHostResultArray(kernelOutputT & h_outputBoards)
        {
            free(h_outputBoards);
            h_outputBoards = nullptr;
        }

        CUDA_HOST void copyResultsArray(kernelOutputT h_outputBoards,
                                        kernelOutputT d_outputBoards,
                                        size_t solversCount)
        {
            cudaError_t err = cudaMemcpy(h_outputBoards,
                                         d_outputBoards,
                                         CUDA_MAX_RESULTS * sizeof(*h_outputBoards),
                                         cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy", err);
            }
        }

        CUDA_HOST void copyResultsArraySize(uint32T * h_outputBoardsSize,
                                            uint32T * d_outputBoardsSize)
        {
            cudaError_t err = cudaMemcpy(h_outputBoardsSize,
                                         d_outputBoardsSize,
                                         sizeof(uint32T),
                                         cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy", err);
            }
        }

        CUDA_HOST void copyCudaEventDevice(std::vector<cudaEventsDeviceT>& h_timers, cudaEventsDeviceT *& d_timers)
        {
            cudaError_t err = cudaMemcpy(h_timers.data(),
                                         d_timers,
                                         h_timers.size() * sizeof(*d_timers),
                                         cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy", err);
            }
        }

        CUDA_HOST bool verifyAllocation(kernelInputT & d_solvers,
                                        kernelOutputT & d_outputBoards,
                                        uint32T* & d_outputBoardsSize)
        {
            return d_solvers != nullptr &&
                d_outputBoards != nullptr &&
                d_outputBoardsSize != nullptr;
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
