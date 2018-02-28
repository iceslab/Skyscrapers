#pragma once
#include <vector>
#include <tuple>
#include <iostream>

class Statistics : public std::vector<std::pair<std::string, double>>
{
public:
    Statistics() = default;
    ~Statistics() = default;

    void print() const;
};

