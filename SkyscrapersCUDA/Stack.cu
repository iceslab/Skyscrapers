#include "Stack.cuh"

namespace cuda
{
    namespace solver
    {
        stackAOST* Stack::allocateAOSStack(uint32T elements)
        {
            stackAOST* d_retVal;

            cudaError_t err = cudaMalloc(&d_retVal, elements * sizeof(stackAOST));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }

            return d_retVal;
        }

        stackSOAT* Stack::allocateSOAStack(uint32T elements)
        {
            stackSOAT h_retVal;
            stackSOAT* d_retVal;

            cudaError_t err = cudaMalloc(&d_retVal, sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation d_retVal", err);
                d_retVal = nullptr;
            }
            else
            {
#ifdef CONTINUOUS_ALLOC
                uint32T* d_block;
                err = cudaMalloc(&d_block, 3 * elements * sizeof(uint32T));
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed allocation d_block", err);
                    d_block = nullptr;
                }

                h_retVal.entry = d_block;
                h_retVal.row = d_block + elements;
                h_retVal.column = d_block + 2 * elements;
#else
                err = cudaMalloc(&h_retVal.entry, elements * sizeof(uint32T));
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed allocation on entry", err);
                    h_retVal.entry = nullptr;
                }
                else
                {
                    err = cudaMalloc(&h_retVal.row, elements * sizeof(uint32T));
                    if (err != cudaSuccess)
                    {
                        CUDA_PRINT_ERROR("Failed allocation on row", err);
                        h_retVal.row = nullptr;
                    }
                    else
                    {
                        err = cudaMalloc(&h_retVal.column, elements * sizeof(uint32T));
                        if (err != cudaSuccess)
                        {
                            CUDA_PRINT_ERROR("Failed allocation on column", err);
                            h_retVal.column = nullptr;
                        }
                    }
                }
#endif
                err = cudaMemcpy(d_retVal, &h_retVal, sizeof(*d_retVal), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memcpy to d_retVal", err);
                    deallocateSOAStackHost(h_retVal);
                }
            }

            return d_retVal;
        }

        void Stack::deallocateAOSStack(stackAOST* & d_stack)
        {
            cudaFree(d_stack);
            d_stack = nullptr;
        }

        void Stack::deallocateSOAStack(stackSOAT* & d_stack)
        {
            stackSOAT h_stack;

            cudaError_t err = cudaMemcpy(&h_stack, d_stack, sizeof(h_stack), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy to d_retVal", err);
            }
            else
            {
                deallocateSOAStackHost(h_stack);
            }
        }

        void Stack::deallocateSOAStackHost(stackSOAT & h_stack)
        {
#ifdef CONTINUOUS_ALLOC
            cudaFree(h_stack.entry);
            h_stack.entry = nullptr;
            h_stack.row = nullptr;
            h_stack.column = nullptr;
#else
            cudaFree(h_stack.entry);
            cudaFree(h_stack.row);
            cudaFree(h_stack.column);
            h_stack.entry = nullptr;
            h_stack.row = nullptr;
            h_stack.column = nullptr;
#endif
        }
    }
}