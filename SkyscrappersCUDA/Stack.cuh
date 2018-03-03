#ifndef __INCLUDED_STACK_CUH__
#define __INCLUDED_STACK_CUH__

#include "CUDAUtilities.cuh"

#define CONTINUOUS_ALLOC

namespace cuda
{
    namespace solver
    {
        typedef struct
        {
            uint32T entry;
            uint32T row;
            uint32T column;
        } stackAOST;

        typedef struct
        {
            uint32T* entry;
            uint32T* row;
            uint32T* column;
        } stackSOAT;


        class Stack
        {
        public:
            Stack() = delete;
            ~Stack() = delete;

            stackAOST* allocateAOSStack(uint32T elements);
            stackSOAT* allocateSOAStack(uint32T elements);
            void deallocateAOSStack(stackAOST* & d_stack);
            void deallocateSOAStack(stackSOAT* & d_stack);
        private:
            void deallocateSOAStackHost(stackSOAT & h_stack);
        };
    }
}

#endif // !__INCLUDED_STACK_CUH__
