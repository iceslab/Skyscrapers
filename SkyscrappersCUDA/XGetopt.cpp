// XGetopt.cpp  Version 1.2
//
// Author:  Hans Dietrich
//          hdietrich2@hotmail.com
//
// Description:
//     XGetopt.cpp implements getopt(), a function to parse command lines.
//
// History
//     Version 1.2 - 2003 May 17
//     - Added Unicode support
//
//     Version 1.1 - 2002 March 10
//     - Added example to XGetopt.cpp module header 
//
// This software is released into the public domain.
// You are free to use it in any way you like.
//
// This software is provided "as is" with no expressed
// or implied warranty.  I accept no liability for any
// damage or loss of business that this software may cause.
//
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// if you are using precompiled headers then include this line:
//#include "stdafx.h"
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// if you are not using precompiled headers then include these lines:
//#include <windows.h>
//#include <stdio.h>
//#include <tchar.h>
///////////////////////////////////////////////////////////////////////////////


#include "XGetopt.h"

///////////////////////////////////////////////////////////////////////////////
//
//  X G e t o p t . c p p
//
//
//  NAME
//       getopt -- parse command line options
//
//  SYNOPSIS
//       int getopt(int argc, TCHAR *argv[], TCHAR *optstring)
//
//       extern TCHAR *optarg;
//       extern int optind;
//
//  DESCRIPTION
//       The getopt() function parses the command line arguments. Its
//       arguments argc and argv are the argument count and array as
//       passed into the application on program invocation.  In the case
//       of Visual C++ programs, argc and argv are available via the
//       variables __argc and __argv (double underscores), respectively.
//       getopt returns the next option letter in argv that matches a
//       letter in optstring.  (Note:  Unicode programs should use
//       __targv instead of __argv.  Also, all character and string
//       literals should be enclosed in _T( ) ).
//
//       optstring is a string of recognized option letters;  if a letter
//       is followed by a colon, the option is expected to have an argument
//       that may or may not be separated from it by white space.  optarg
//       is set to point to the start of the option argument on return from
//       getopt.
//
//       Option letters may be combined, e.g., "-ab" is equivalent to
//       "-a -b".  Option letters are case sensitive.
//
//       getopt places in the external variable optind the argv index
//       of the next argument to be processed.  optind is initialized
//       to 0 before the first call to getopt.
//
//       When all options have been processed (i.e., up to the first
//       non-option argument), getopt returns EOF, optarg will point
//       to the argument, and optind will be set to the argv index of
//       the argument.  If there are no non-option arguments, optarg
//       will be set to NULL.
//
//       The special option "--" may be used to delimit the end of the
//       options;  EOF will be returned, and "--" (and everything after it)
//       will be skipped.
//
//  RETURN VALUE
//       For option letters contained in the string optstring, getopt
//       will return the option letter.  getopt returns a question mark (?)
//       when it encounters an option letter not included in optstring.
//       EOF is returned when processing is finished.
//
//  BUGS
//       1)  Long options are not supported.
//       2)  The GNU double-colon extension is not supported.
//       3)  The environment variable POSIXLY_CORRECT is not supported.
//       4)  The + syntax is not supported.
//       5)  The automatic permutation of arguments is not supported.
//       6)  This implementation of getopt() returns EOF if an error is
//           encountered, instead of -1 as the latest standard requires.
//
//  EXAMPLE
//       BOOL CMyApp::ProcessCommandLine(int argc, TCHAR *argv[])
//       {
//           int c;
//
//           while ((c = getopt(argc, argv, _T("aBn:"))) != EOF)
//           {
//               switch (c)
//               {
//                   case _T('a'):
//                       TRACE(_T("option a\n"));
//                       //
//                       // set some flag here
//                       //
//                       break;
//
//                   case _T('B'):
//                       TRACE( _T("option B\n"));
//                       //
//                       // set some other flag here
//                       //
//                       break;
//
//                   case _T('n'):
//                       TRACE(_T("option n: value=%d\n"), atoi(optarg));
//                       //
//                       // do something with value here
//                       //
//                       break;
//
//                   case _T('?'):
//                       TRACE(_T("ERROR: illegal option %s\n"), argv[optind-1]);
//                       return FALSE;
//                       break;
//
//                   default:
//                       TRACE(_T("WARNING: no handler for option %c\n"), c);
//                       return FALSE;
//                       break;
//               }
//           }
//           //
//           // check for non-option args here
//           //
//           return TRUE;
//       }
//
///////////////////////////////////////////////////////////////////////////////

TCHAR	*optarg;		// global argument pointer
int		optind = 0; 	// global argv index

bool loadFromFile;
const char* filePath;
std::vector<bool> solversEnabled(SOLVERS_SIZE, false);
size_t gpuAlgorithmsToRun;
size_t boardDimension = 1;
size_t desiredBoards = 1;

int getopt(int argc, TCHAR *argv[], TCHAR *optstring)
{
    static TCHAR *next = NULL;
    if (optind == 0)
        next = NULL;

    optarg = NULL;

    if (next == NULL || *next == _T('\0'))
    {
        if (optind == 0)
            optind++;

        if (optind >= argc || argv[optind][0] != _T('-') || argv[optind][1] == _T('\0'))
        {
            optarg = NULL;
            if (optind < argc)
                optarg = argv[optind];
            return EOF;
        }

        if (_tcscmp(argv[optind], _T("--")) == 0)
        {
            optind++;
            optarg = NULL;
            if (optind < argc)
                optarg = argv[optind];
            return EOF;
        }

        next = argv[optind];
        next++;		// skip past -
        optind++;
    }

    TCHAR c = *next++;
    TCHAR *cp = _tcschr(optstring, c);

    if (cp == NULL || c == _T(':'))
        return _T('?');

    cp++;
    if (*cp == _T(':'))
    {
        if (*next != _T('\0'))
        {
            optarg = next;
            next = NULL;
        }
        else if (optind < argc)
        {
            optarg = argv[optind];
            optind++;
        }
        else
        {
            return _T('?');
        }
    }

    return c;
}

bool ProcessCommandLine(int argc, TCHAR *argv[])
{
    int c;

    while ((c = getopt(argc, argv, _T("f:scg:d:b:p:h"))) != EOF)
    {
        switch (c)
        {
        case _T('f'):
            loadFromFile = true;
            filePath = optarg;
            break;
        case _T('s'):
            solversEnabled[SEQUENTIAL] = true;
            break;
        case _T('c'):
            solversEnabled[PARALLEL_CPU] = true;
            break;
        case _T('g'):
            parseGPUOptarg(optarg);
            break;
        case _T('d'):
            boardDimension = std::stoull(optarg);
            boardDimension = boardDimension < 1 ? 1 : boardDimension;
            break;
        case _T('b'):
            desiredBoards = std::stoull(optarg);
            desiredBoards = desiredBoards < 1 ? 1 : desiredBoards;
            break;
        case _T('p'):
            desiredFifoSize = std::stoull(optarg);
            desiredFifoSize = desiredFifoSize < 1 ? 1 : desiredFifoSize;
            break;
        case _T('h'):
            return FALSE;
            break;
        case _T('?'):
            printf("ERROR: illegal option %s\n", argv[optind - 1]);
            return FALSE;
            break;
        default:
            printUsage();
            printf("WARNING: no handler for option %c\n", c);
            return FALSE;
            break;
        }
    }

    //
    // check for non-option args here
    //
    return TRUE;
}

void printUsage()
{
    printf("SkyscrappersCUDA usage:\n\n");

    printf("   -f path\n"
           "      Loads file from given path. Overrides -b option\n");
    printf("   -s\n"
           "      Program will run sequential algorithm\n");
    printf("   -c\n"
           "      Program will run parallel algorithm on CPU\n");
    printf("   -g algorithm\n"
           "      Program will run chosen algorithm(s) on GPU\n"
           "      Valid options are: \"all\", \"basic\", \"inc\", \"aos\" and \"soa\"\n"
           "      \"all\"   - run all available algorithms\n"
           "      \"basic\" - run basic algorithm\n"
           "      \"inc\"   - run incremental stack algorithm\n"
           "      \"aos\"   - run Array of Structures stack algorithm\n"
           "      \"soa\"   - run Structure of Arrays stack algorithm\n");
    printf("   -d dimension\n"
           "      Dimensions of generated square board\n");
    printf("   -b boards to generate\n"
           "      Number of boards wchich program will generate when running parallel algorithm.\n"
           "      It determines number of launched threads. Option ignored when used with -f\n");
    printf("   -p FIFO size in bytes\n"
           "      Determines CUDA FIFO size in bytes. Useful when debugging\n"
           "      FIFO is used to store device's printf output during kernel execution.\n"
           "      Default value is 1MB (1 048 576 bytes)\n");
    printf("   -h\n"
           "      Prints this help text\n");
}

void printLaunchParameters()
{
    if (loadFromFile)
    {
        printf("Loading board from: \"%s\"\n", filePath);
    }
    else
    {
        printf("Generating %zux%zu board\n", boardDimension, boardDimension);
    }

    printf("Number of threads in parallel algorithms: %zu\n", desiredBoards);
    printf("Sequential algorithm:                     %s\n", boolToEnabled(solversEnabled[SEQUENTIAL]));
    printf("Parallel CPU algorithm:                   %s\n", boolToEnabled(solversEnabled[PARALLEL_CPU]));
    printf("Base parallel GPU algorithm:              %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_BASE]));
    printf("Incremental stack parallel GPU algorithm: %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_INCREMENTAL]));
    printf("AoS stack parallel GPU algorithm:         %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_AOS]));
    printf("SoA stack parallel GPU algorithm:         %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_SOA]));
}

const char * boolToEnabled(bool option)
{
    return option ? "enabled" : "disabled";
}

void parseGPUOptarg(const std::string & optarg)
{
    std::vector<std::string> substrings = { "all", "basic", "inc", "aos", "soa" };

    for (size_t i = 0; i < substrings.size(); i++)
    {
        const auto & el = substrings[i];
        if (optarg.find(el) == std::string::npos)
        {
            continue;
        }

        switch (i)
        {
        case 0:
            for (size_t i = PARALLEL_GPU_BEGIN; i < PARALLEL_GPU_END; i++)
            {
                solversEnabled[i] = true;
            }
            return;
        case 1:
            solversEnabled[PARALLEL_GPU_BASE] = true;
            break;
        case 2:
            solversEnabled[PARALLEL_GPU_INCREMENTAL] = true;
            break;
        case 3:
            solversEnabled[PARALLEL_GPU_AOS] = true;
            break;
        case 4:
            solversEnabled[PARALLEL_GPU_SOA] = true;
            break;
        default:
            printf("Error: Substring has no match!\n");
            break;
        }
    }

    gpuAlgorithmsToRun = std::count(solversEnabled.begin(), solversEnabled.end(), true);
}
