#pragma once
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include "CUDAUtilities.cuh"

#define RESULT_DELIMITER ";"

class Statistics : public std::vector<std::tuple<std::string, std::string, double>>
{
public:
    Statistics(bool printable = true);
    ~Statistics() = default;

    void emplace_back(std::string shortName, std::string longName, double timeInMilliseconds);
    void emplace_back(std::string longName, double timeInMilliseconds);

    bool printable;
    void print() const;
    void writeToFile(std::string path, bool printHeaders = false) const;
};

