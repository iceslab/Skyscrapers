#pragma once
#include <cstdio>
#include <climits>
#include <cassert>

#define QUOTE(x) #x
#define ASSERT(expr) assert(expr);

#ifdef _DEBUG
#define DEBUG_PRINTLN_VERBOSE(format, ...) \
do{ \
	fprintf(stderr, format, __VA_ARGS__); \
	fprintf(stderr, ", file %s, line %d\n", __FILE__, __LINE__); \
} while (false);

#define DEBUG_PRINT_VERBOSE(format, ...) \
do{ \
	fprintf(stderr, format, __VA_ARGS__); \
	fprintf(stderr, ", file %s, line %d", __FILE__, __LINE__); \
} while (false);

#define DEBUG_PRINTLN(format, ...) \
do{ \
	fprintf(stderr, format"\n", __VA_ARGS__); \
} while (false);

#define DEBUG_PRINT(format, ...) \
do{ \
	fprintf(stderr, format, __VA_ARGS__); \
} while (false);

#define DEBUG_CALL(expr) \
do{ \
	expr; \
} while (false);

#define ASSERT_VERBOSE(expr, format, ...) \
do{ \
	fprintf(stderr, "Assertion failed: " QUOTE(expr) ", "); \
	fprintf(stderr, format, __VA_ARGS__); \
	fprintf(stderr, ", file %s, line %d\n", __FILE__, __LINE__); \
} while (false);
#else
#define DEBUG_PRINTLN_VERBOSE(format, ...)
#define DEBUG_PRINT_VERBOSE(format, ...)

#define DEBUG_PRINTLN(format, ...)
#define DEBUG_PRINT(format, ...)

#define DEBUG_CALL(expr)

#define ASSERT_VERBOSE(expr, format, ...)
#endif

#define MAX_FACTORIAL_64 20
#define MAX_FACTORIAL_32 12
#define MAX_FACTORIAL_16 8
#define MAX_FACTORIAL_8 5

#define MAX_FACTORIAL_EXCEEDED_MSG "Result of factorial exceeds max storage value for return type"