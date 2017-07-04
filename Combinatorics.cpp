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
		// (n)      n(n-1)...(n-k+1)
		// ( ) = ---------------------
		// (k)        1*2*...*k

		auto numerator = n;
		auto denominator = 1;
		for (size_t i = 2; i < k; i++)
		{
			numerator *= n - i + 1;
			denominator *= i;
		}

		// It will always be integer
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

void Combinatorics::generateAllPermutations(std::vector<int> A)
{
	// Based on Heap's algorithm
	int n = A.size();
	std::vector<int> c(n);
	std::copy(A.begin(), A.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	int i = 0;
	while (i < n)
	{
		if (c[i] < i)
		{
			// Is odd
			if (i & 1)
			{
				std::iter_swap(A.begin() + c[i], A.begin() + i);
			}
			// Is even
			else
			{
				std::iter_swap(A.begin(), A.begin() + i);
			}

			std::copy(A.begin(), A.end(), std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;
			c[i]++;
			i = 0;
		}
		else
		{
			c[i] = 0;
			i++;
		}
	}
}