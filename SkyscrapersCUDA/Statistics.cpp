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
            auto time = cuda::timeToHumanReadable(el.second, cuda::MILLISECONDS);
            printf("%s%.2f %s\n", el.first.c_str(), time.first, time.second.c_str());
        }
    }
}
