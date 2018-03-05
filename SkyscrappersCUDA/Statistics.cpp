#include "Statistics.h"

Statistics::Statistics(bool printable) : printable(printable)
{
    // Nothing to do
}

void Statistics::print() const
{
    if (printable)
    {
        for (const auto & el : *this)
        {
            std::cout << el.first.c_str() << el.second << " ms" << std::endl;
        }
    }
}
