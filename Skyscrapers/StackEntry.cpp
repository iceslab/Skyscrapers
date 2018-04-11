#include "StackEntry.h"

namespace solver
{
    const size_t StackEntry::badIndex = std::numeric_limits<size_t>::max();
    
    StackEntry::StackEntry(const size_t size) : std::vector<bool>(size), setBitsCount(0)
    {
        // Nothing to do
    }

    bool StackEntry::all() const
    {
        return size() == setBitsCount;
    }

    bool StackEntry::any() const
    {
        return setBitsCount > 0;
    }

    bool StackEntry::none() const
    {
        return setBitsCount == 0;
    }

    bool StackEntry::getBit(const size_t pos) const
    {
        return (*this)[pos];
    }

    void StackEntry::setBit(const size_t pos, bool val)
    {
        if ((*this)[pos] != val)
        {
            if ((*this)[pos])
            {
                --setBitsCount;
            }
            else
            {
                ++setBitsCount;
            }
            (*this)[pos] = val;
        }
    }

    void StackEntry::resetBit(const size_t pos)
    {
        setBit(pos, false);
    }

    void StackEntry::flipBit(const size_t pos)
    {
        if ((*this)[pos])
        {
            --setBitsCount;
        }
        else
        {
            ++setBitsCount;
        }
        (*this)[pos] = !(*this)[pos];
    }
    
    size_t StackEntry::firstZero() const
    {
        auto retVal = badIndex;
        if (!all())
        {
            const auto it = std::find(this->begin(), this->end(), false);
            if (it != this->end())
            {
                retVal = it - begin();
            }
        }
        return retVal;
    }
}
