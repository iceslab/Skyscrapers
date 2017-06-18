#pragma once
#include <concurrent_unordered_map.h>
class Combinatorics
{
public:
	static size_t factorial(size_t n);

private:
	
	static concurrency::concurrent_unordered_map<size_t, size_t> factorialMemoization;
	Combinatorics() = delete;
	~Combinatorics() = delete;

	static decltype(factorialMemoization) initialize();
};

