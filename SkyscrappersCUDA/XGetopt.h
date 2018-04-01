// XGetopt.h  Version 1.2
//
// Author:  Hans Dietrich
//          hdietrich2@hotmail.com
//
// This software is released into the public domain.
// You are free to use it in any way you like.
//
// This software is provided "as is" with no expressed
// or implied warranty.  I accept no liability for any
// damage or loss of business that this software may cause.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef XGETOPT_H
#define XGETOPT_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include <string>
#include <vector>

#define CUDA_MAX_THREADS_IN_BLOCK (1024)
#define CUDA_MAX_BLOCKS_OF_THREADS (1024)

extern int optind, opterr;
extern TCHAR *optarg;

enum SolversEnableE
{
    SEQUENTIAL = 0,
    PARALLEL_CPU,
    PARALLEL_GPU_BEGIN,
    PARALLEL_GPU_BASE = PARALLEL_GPU_BEGIN,
    PARALLEL_GPU_INCREMENTAL,
    PARALLEL_GPU_SHM,
    PARALLEL_GPU_AOS,
    PARALLEL_GPU_SOA,
    PARALLEL_GPU_END,
    SOLVERS_SIZE = PARALLEL_GPU_END
};

extern bool loadFromFile;
extern const char* filePath;
extern std::vector<bool> solversEnabled;
extern size_t gpuAlgorithmsToRun;
extern size_t boardDimension;
extern size_t threadsInBlock;
extern size_t blocksOfThreads;
extern size_t desiredBoards;
extern size_t desiredFifoSize;

int getopt(int argc, TCHAR *argv[], TCHAR *optstring);

bool ProcessCommandLine(int argc, TCHAR *argv[]);

void printUsage();
void printLaunchParameters();
const char* boolToEnabled(bool option);
void parseGPUOptarg(const std::string & optarg);
const char* enumToKernelName(SolversEnableE solverType);
const char* enumToSolverName(SolversEnableE solverType);

#endif //XGETOPT_H
