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

extern int optind, opterr;
extern TCHAR *optarg;

extern bool loadFromFile;
extern const char* filePath;
extern bool sequentialSolver;
extern bool parallelCpuSolver;
extern bool parallelGpuSolver;
extern size_t boardDimension;
extern size_t desiredBoards;
extern size_t desiredFifoSize;

int getopt(int argc, TCHAR *argv[], TCHAR *optstring);

bool ProcessCommandLine(int argc, TCHAR *argv[]);

#endif //XGETOPT_H
