#include "Timer.h"

// Strange two lines of code are here, because of two reasons
// First, all static members must be initialized
// Second, in C++ there is no guarantee of initialization order,
// therefore there should not be frequency = {0},
// frequency = frequency, guarantees that we have valid value in frequency
#ifndef USE_CHRONO
LARGE_INTEGER Timer::frequency = frequency;
BOOL Timer::isFrequencyInitialized = QueryPerformanceFrequency(&frequency);
#endif


Timer::Timer()
{
#ifndef USE_CHRONO
	memset(&pcStart, 0, sizeof(pcStart));
	memset(&pcStop, 0, sizeof(pcStop));
#endif
}

void Timer::start()
{
#ifdef USE_CHRONO
	chStart = std::chrono::high_resolution_clock::now();
	chStop = chStart;
#else
	QueryPerformanceCounter(&pcStart);
	memcpy(&pcStop, &pcStart, sizeof(pcStart));
#endif // USE_CHRONO
}

double Timer::stop(Resolution resolution)
{
#ifdef USE_CHRONO
	chStop = std::chrono::high_resolution_clock::now();
#else
	QueryPerformanceCounter(&pcStop);
#endif // USE_CHRONO
	return getTime();
}

double Timer::getTime(Resolution resolution)
{
	auto duration = 0.0;
#ifdef USE_CHRONO
	std::chrono::duration<double, std::nano> elapsedNano = chStop - chStart;
	duration = elapsedNano.count();
#else
	LARGE_INTEGER elapsedNano;

	elapsedNano.QuadPart = pcStop.QuadPart - pcStart.QuadPart;
	elapsedNano.QuadPart *= SECONDS;
	duration = static_cast<double>(elapsedNano.QuadPart) / static_cast<double>(frequency.QuadPart);
#endif // USE_CHRONO
	return convertResolution(NANOSECONDS, resolution, duration);
}

double Timer::convertResolution(Resolution from, Resolution to, double time)
{
#ifdef _DEBUG
	auto d = time * (static_cast<double>(from) / static_cast<double>(to));
#endif
	return time * (static_cast<double>(from) / static_cast<double>(to));
}
