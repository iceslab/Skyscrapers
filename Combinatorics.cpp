#include "Combinatorics.h"

factorialMapT Combinatorics::factorialMemoization = Combinatorics::initialize();

factorialT Combinatorics::factorial(factorialT n)
{
	if (n > determineMaxFactorial<factorialT>())
	{
		DEBUG_PRINT(MAX_FACTORIAL_EXCEEDED_MSG);
		throw std::invalid_argument(MAX_FACTORIAL_EXCEEDED_MSG);
	}

	auto memoized = factorialMemoization.find(n);
	if (memoized != factorialMemoization.end())
	{
		return memoized->second;
	}
	else
	{
		auto retVal = factorial(n - 1) * n;
		factorialMemoization.insert(std::make_pair(n, retVal));
		return retVal;
	}
}

factorialT Combinatorics::binomialExpansion(factorialT n, factorialT k)
{
	if (k > n)
	{
		return 0;
	}
	else if (k == 0 || k == n)
	{
		return 1;
	}
	else if (k == 1)
	{
		return n;
	}
	else
	{
		// (n)      n!
		// ( ) = ---------
		// (k)    k!(n-k)!

		auto numerator = factorial(n);
		auto denominator = factorial(k) * factorial(n - k);
		return numerator / denominator;
	}
}

factorialMapT Combinatorics::initialize()
{
	factorialMapT retVal;

	if (PREMEMOIZE_N_FACTORIAL > determineMaxFactorial<factorialT>())
	{	
		DEBUG_PRINT(MAX_FACTORIAL_EXCEEDED_MSG);
		throw std::invalid_argument(MAX_FACTORIAL_EXCEEDED_MSG);
	}

	retVal.insert(std::make_pair(0, 1));
	retVal.insert(std::make_pair(1, 1));

	for (size_t i = 2, count = 2; i < PREMEMOIZE_N_FACTORIAL; i++, count *= i)
	{
		retVal.insert(std::make_pair(i, count));
	}

	return retVal;
}
