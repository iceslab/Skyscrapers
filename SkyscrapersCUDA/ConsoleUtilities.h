#ifndef __INCLUDED_CONSOLE_UTILITIES_H__
#define __INCLUDED_CONSOLE_UTILITIES_H__

#include <stdio.h>
#include <algorithm>
#include "Statistics.h"

#ifndef _WIN32
    #include <unistd.h>
    #include <getopt.h>
    #define TCHAR char
#else
    #include "getopt.h"
#endif // !_WIN32

#define MAX_PWD_PATH 1024

extern bool loadFromFile;
extern const char* filePath;
extern std::vector<bool> solversEnabled;
extern size_t gpuAlgorithmsToRun;
extern size_t boardDimension;
extern size_t threadsInBlock;
extern size_t blocksOfThreads;
extern size_t desiredBoards;
extern size_t desiredFifoSize;
extern bool resultsToFile;
extern const char* resultsPath;
extern bool muteOutput;
extern bool columnHeaders;

// Command line processing functions
bool ProcessCommandLine(int argc, char *argv[]);
void printUsage();
void printLaunchParameters();
const char *boolToEnabled(bool option);
void parseGPUOptarg(const std::string &optarg);
const char *enumToKernelName(SolversEnableE solverType);
const char *enumToSolverName(SolversEnableE solverType);
void getCurrentDirectory(TCHAR *buffer, size_t size);
void printPwd();
void printf_m(const char *format, ...);

#endif // !__INCLUDED_CONSOLE_UTILITIES_H__
