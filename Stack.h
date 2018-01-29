#pragma once
#include <cstdint>
#include <amp.h>
#include "Pair.h"

namespace AMP
{

    class Stack
    {
    public:
        typedef uint32_t leftEntryT;
        typedef Pair<size_t, size_t> rightEntryT;
        typedef Pair<leftEntryT, rightEntryT> stackEntryT;

        Stack(const size_t maxSize) __CPU_ONLY;
        ~Stack() __CPU_ONLY;

        /// Mutators
        void emplaceBack(const leftEntryT & first, const rightEntryT & second) __GPU;
        void popBack() __GPU;

        /// Accessors
        stackEntryT back() __GPU;

        /// Container info
        bool empty() const __GPU;
        size_t size() const __GPU;
    private:
        const size_t maxSize;
        size_t currentSize;

        Concurrency::array<stackEntryT, 1> stack;
    };
}
