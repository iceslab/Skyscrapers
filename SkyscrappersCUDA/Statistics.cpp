#include "Statistics.h"

void Statistics::print() const
{
    for (const auto & el : *this)
    {
        std::cout << el.first.c_str() << el.second << " ms" << std::endl;
    }
}
