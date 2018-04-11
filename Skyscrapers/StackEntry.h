#ifndef __INCLUDED_STACK_ENTRY_H__
#define __INCLUDED_STACK_ENTRY_H__
#include <cstdlib>
#include <limits>
#include <vector>
#include <algorithm>

namespace solver
{
    class StackEntry : private std::vector<bool>
    {
    public:
        StackEntry() = default;
        StackEntry(const size_t size);
        ~StackEntry() = default;

        static const size_t badIndex;

        bool all() const;
        bool any() const;
        bool none() const;

        bool getBit(const size_t pos) const;
        void setBit(const size_t pos, bool val = true);
        void resetBit(const size_t pos);
        void flipBit(const size_t pos);

        size_t firstZero() const;
    private:
        size_t setBitsCount;
        
    };
}
#endif //!__INCLUDED_STACK_ENTRY_H__
