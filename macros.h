#pragma once
#include <cstdio>
#include <climits>

#ifdef _DEBUG
#define DEBUG_PRINT(format, ...) \
do{ \
	fprintf(stderr, "File: %s(%d):\n\tMsg: ", __FILE__, __LINE__); \
	fprintf(stderr, format"\n", __VA_ARGS__); \
} while (false);
#else
#define DEBUG_PRINT(format, ...)
#endif

#define MAX_FACTORIAL_64 20
#define MAX_FACTORIAL_32 12
#define MAX_FACTORIAL_16 8
#define MAX_FACTORIAL_8 5

#define MAX_FACTORIAL_EXCEEDED_MSG "Result of factorial exceeds max storage value for return type"