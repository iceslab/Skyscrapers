#include "Stack.h"

namespace AMP
{

    Stack::Stack(const size_t maxSize) __CPU_ONLY : maxSize(maxSize), stack(maxSize)
    {
        // Nothing to do
    }

    Stack::~Stack() __CPU_ONLY
    {
        // Nothing to do
    }

    void Stack::emplaceBack(const leftEntryT & first, const rightEntryT & second) __GPU
    {
        if (currentSize < maxSize)
        {
            ++currentSize;
            stack[currentSize].first = first;
            stack[currentSize].second = second;
        }
        else
        {
            // Nothing to do            
        }
    }

    void Stack::popBack() __GPU
    {
        --currentSize;
    }

    Stack::stackEntryT Stack::back() __GPU
    {
        return stack[currentSize - 1];
    }

    bool Stack::empty() const __GPU
    {
        return currentSize == 0;
    }

    size_t Stack::size() const __GPU
    {
        return currentSize;
    }
}
