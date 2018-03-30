#include "Solver.cuh"

namespace cuda
{
    namespace solver
    {
        Solver::Solver(const board::Board & board) : board(board)
        {
            // Nothing to do
        }

        Solver::Solver(board::Board && board) : board(board)
        {
            // Nothing to do
        }

        CUDA_HOST_DEVICE Solver::Solver(const cuda::Board& board,
                       void * constantMemoryPtr,
                       void * sharedMemoryPtr) :
            board(board, constantMemoryPtr, sharedMemoryPtr)
        {
            // Nothing to do
        }

        CUDA_HOST_DEVICE Solver::~Solver()
        {
            // Nothing to do
        }
    }
}
