#pragma once
#include <vector>

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
