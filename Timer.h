#pragma once
//#define USE_CHRONO
#ifdef USE_CHRONO
#include <chrono>
#else
#include <Windows.h>
#endif // USE_CHRONO

#include <cmath>

enum Resolution
{
	NANOSECONDS = 1,
	MICROSECONDS = NANOSECONDS * 1000,
	MILLISECONDS = MICROSECONDS * 1000,
	SECONDS = MILLISECONDS * 1000
};

class Timer
{
public:
	Timer();

	void start();
	double stop(Resolution resolution = SECONDS);
	double getTime(Resolution resolution = SECONDS);

	static double convertResolution(Resolution from, Resolution to, double time);
private:
#ifdef USE_CHRONO
	std::chrono::time_point<std::chrono::high_resolution_clock> chStart;
	std::chrono::time_point<std::chrono::high_resolution_clock> chStop;
#else
	LARGE_INTEGER pcStart;
	LARGE_INTEGER pcStop;
	static LARGE_INTEGER frequency;
	static BOOL isFrequencyInitialized;
#endif
};

