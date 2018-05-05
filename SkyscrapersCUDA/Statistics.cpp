#include "Statistics.h"

Statistics::Statistics(bool printable) : printable(printable)
{
    // Nothing to do
}

void Statistics::emplace_back(std::string shortName,
                              std::string longName,
                              double timeInMilliseconds)
{
    this->push_back(std::make_tuple(shortName, longName, timeInMilliseconds));
}

void Statistics::emplace_back(std::string longName, double timeInMilliseconds)
{
    this->emplace_back(longName, longName, timeInMilliseconds);
}

void Statistics::print() const
{
    if (printable)
    {
        for (const auto & el : *this)
        {
          const auto &longName = std::get<1>(el);
          auto time = cuda::timeToHumanReadable(std::get<2>(el), cuda::MILLISECONDS);
          printf("%s%.2f %s\n", longName.c_str(), time.first,
                 time.second.c_str());
        }
    }
}

void Statistics::writeToFile(std::string path, bool printHeaders) const
{
    if (printable)
    {
        std::ofstream ofs(path);

        if(ofs.is_open())
        {
            if(printHeaders)
            {
                // Print column headers
                for (size_t i = 0; i < this->size(); i++)
                {
                    const auto &shortName = std::get<0>((*this)[i]);
                    ofs << shortName;

                    if(i < this->size() - 1)
                    {
                        ofs << RESULT_DELIMITER;
                    }
                }
                ofs << "\n";
            }

            for (size_t i = 0; i < this->size(); i++)
            {
                const auto &time = std::get<2>((*this)[i]);
                ofs << time;

                if(i < this->size() - 1)
                {
                    ofs << RESULT_DELIMITER;
                }
            }
            ofs << "\n";
        }
        else
        {
            printf("Could not save to file \"%s\"", path.c_str());
        }
    }
}
