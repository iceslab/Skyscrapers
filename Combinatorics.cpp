#include "Combinatorics.h"

concurrency::concurrent_unordered_map<size_t, size_t> Combinatorics::factorialMemoization = Combinatorics::initialize();

size_t Combinatorics::factorial(size_t n)
{
	auto memoized = factorialMemoization.find(n);
	if (memoized != factorialMemoization.end())
	{
		return memoized->second;
	}
	else
	{
		auto retVal = factorial(n - 1) * n;
		factorialMemoization.insert(retVal);
		return retVal;
	}
}

decltype(Combinatorics::factorialMemoization) Combinatorics::initialize()
{
	decltype(factorialMemoization) retVal;
	retVal.insert(0, 1);
	retVal.insert(1, 1);
	return retVal;
}
