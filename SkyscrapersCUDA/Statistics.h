#pragma once
#include <vector>
#include <tuple>
#include <iostream>
#include "CUDAUtilities.cuh"

class Statistics : public std::vector<std::pair<std::string, double>>
{
public:
    Statistics(bool printable = true);
    ~Statistics() = default;

    bool printable;
    void print() const;
};

