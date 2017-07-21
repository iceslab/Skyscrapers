#pragma once
#include <concurrent_unordered_map.h>
#include "macros.h"
#include <stdexcept>
#include <vector>

#include <algorithm>
#include <iterator>
#include <iostream>

#define PREMEMOIZE_N_FACTORIAL 20
#ifndef PREMEMOIZE_N_FACTORIAL
#define PREMEMOIZE_N_FACTORIAL 2U
#endif

typedef uint64_t factorialT;
typedef concurrency::concurrent_unordered_map<factorialT, factorialT> factorialMapT;

class Combinatorics
{
public:
    static factorialT factorial(factorialT n);
    static factorialT binomialExpansion(factorialT n, factorialT k);
    static void generateAllPermutations(std::vector<int> A);
private:

    static factorialMapT factorialMemoization;
    static bool memoizationInit;
    Combinatorics() = delete;
    ~Combinatorics() = delete;

    static factorialMapT initialize();
    template<typename T> static constexpr factorialT determineMaxFactorial();
};

template<typename T>
inline constexpr factorialT Combinatorics::determineMaxFactorial()
{
    if (!std::is_integral<T>() || !std::is_unsigned<T>())
        return 0;

    auto typeSize = sizeof(T) * CHAR_BIT;
    switch (typeSize)
    {
        case 64:
            return MAX_FACTORIAL_64;
        case 32:
            return MAX_FACTORIAL_32;
        case 16:
            return MAX_FACTORIAL_16;
        case 8:
            return MAX_FACTORIAL_8;
        default:
            return 0;
    }
}
