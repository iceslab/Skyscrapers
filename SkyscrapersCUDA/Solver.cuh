#ifndef __INCLUDED_SOLVER_CUH__
#define __INCLUDED_SOLVER_CUH__
#include "../Skyscrapers/Board.h"
#include "Board.cuh"

namespace cuda
{
    namespace solver
    {
        class Solver
        {
        public:
            Solver(const board::Board& board);
            Solver(board::Board&& board);
            CUDA_HOST_DEVICE Solver(const cuda::Board& board,
                                    void * constantMemoryPtr = nullptr,
                                    void * sharedMemoryPtr = nullptr);
            CUDA_HOST_DEVICE ~Solver();

            cuda::Board board;
        };
    }
}
#endif // !__INCLUDED_SOLVER_CUH__
