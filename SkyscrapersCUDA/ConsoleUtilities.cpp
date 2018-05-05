#include "ConsoleUtilities.h"

bool loadFromFile = false;
const char* filePath = "";
std::vector<bool> solversEnabled(SOLVERS_SIZE, false);
size_t gpuAlgorithmsToRun = 0;
size_t boardDimension = 1;
size_t threadsInBlock = 1;
size_t blocksOfThreads = 1;
size_t desiredBoards = 1;
size_t desiredFifoSize = CUDA_DEFAULT_FIFO_SIZE;
bool resultsToFile = false;
const char* resultsPath = "";
bool muteOutput = false;
bool columnHeaders = false;

bool ProcessCommandLine(int argc, char *argv[])
{
    int c;

    while ((c = getopt(argc, argv, "f:scg:d:t:b:p:q:mr:vh")) != EOF)
    {
        switch (c)
        {
        case 'f':
            loadFromFile = true;
            filePath = optarg;
            break;
        case 's':
            solversEnabled[SEQUENTIAL] = true;
            break;
        case 'c':
            solversEnabled[PARALLEL_CPU] = true;
            break;
        case 'g':
            parseGPUOptarg(optarg);
            break;
        case 'd':
            boardDimension = std::stoull(optarg);
            boardDimension = boardDimension < 1 ? 1 : boardDimension;
            break;
        case 't':
            threadsInBlock = std::stoull(optarg);
            threadsInBlock = threadsInBlock < 1 ? 1 : threadsInBlock % (CUDA_MAX_THREADS_IN_BLOCK + 1);
            break;
        case 'b':
            blocksOfThreads = std::stoull(optarg);
            blocksOfThreads = blocksOfThreads < 1 ? 1 : blocksOfThreads % (CUDA_MAX_BLOCKS_OF_THREADS + 1);
            break;
        case 'p':
            desiredBoards = std::stoull(optarg);
            desiredBoards = desiredBoards < 1 ? 1 : desiredBoards;
            break;
        case 'q':
            desiredFifoSize = std::stoull(optarg);
            desiredFifoSize = desiredFifoSize < 1 ? 1 : desiredFifoSize;
            break;
        case 'm':
            muteOutput = true;
            break;
        case 'r':
            resultsToFile = true;
            resultsPath = optarg;
            break;
        case 'v':
            columnHeaders = true;
            break;
        case 'h':
            return false;
        case '?':
            printf("ERROR: illegal option %s\n", argv[optind - 1]);
            return false;
        default:
            printUsage();
            printf("WARNING: no handler for option %c\n", c);
            return false;
        }
    }

    //
    // check for non-option args here
    //
    return true;
}

void printUsage()
{
    // Intentionally ignoring muteOutput option
    printf("SkyscrappersCUDA usage:\n\n");

    printf("   -f path\n"
           "      Loads file from given path. Overrides -p option\n");
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
           "      \"shm\"   - run shared memory algorithm\n"
           "      \"aos\"   - run Array of Structures stack algorithm\n"
           "      \"soa\"   - run Structure of Arrays stack algorithm\n");
    printf("   -d dimension\n"
           "      Dimensions of generated square board\n");
    printf("   -t threads per block\n"
           "      Number of threads in block used in parallelizing algorithm.\n"
           "      Maximum allowed is %d\n",
           CUDA_MAX_THREADS_IN_BLOCK);
    printf("   -b blocks of threads\n"
           "      Number of blocks used in parallelizing algorithm.\n"
           "      Maximum allowed is %d\n",
           CUDA_MAX_BLOCKS_OF_THREADS);
    printf("   -p boards to generate\n"
           "      Number of boards which program will generate when running parallel algorithm.\n"
           "      Also determines number of threads in parallel CPU algorithms\n");
    printf("   -q FIFO size in bytes\n"
           "      Determines CUDA FIFO size in bytes. Useful when debugging\n"
           "      FIFO is used to store device's printf output during kernel execution.\n"
           "      Default value is 1MB (1 048 576 bytes)\n");
    printf("   -m\n"
           "      Silent mode. No output will be printed.\n");
    printf("   -r path\n"
           "      Saves results to specified file. Best used with -m switch.\n");
    printf("   -v\n"
           "      Print column headers in results file. No effect without -r switch.\n");
    printf("   -h\n"
           "      Prints this help text\n");
}

void printLaunchParameters()
{
    if (loadFromFile)
    {
        printf_m("Loading board from: \"%s\"\n", filePath);
    }
    else
    {
        printf_m("Generating %zux%zu board\n", boardDimension, boardDimension);
    }

    printf_m("Number of threads in parallel CPU algorithms: %s\n", "equal to generated boards");
    printf_m("Number of threads in parallel GPU algorithms: %7zu\n", threadsInBlock * blocksOfThreads);
    printf_m("Number of threads per block:                  %7zu\n", threadsInBlock);
    printf_m("Number of blocks of threads:                  %7zu\n", blocksOfThreads);
    printf_m("Sequential algorithm:                     %s\n", boolToEnabled(solversEnabled[SEQUENTIAL]));
    printf_m("Parallel CPU algorithm:                   %s\n", boolToEnabled(solversEnabled[PARALLEL_CPU]));
    printf_m("Base parallel GPU algorithm:              %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_BASE]));
    printf_m("Incremental stack parallel GPU algorithm: %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_INCREMENTAL]));
    printf_m("Shared memory parallel GPU algorithm:     %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_SHM]));
    printf_m("AoS stack parallel GPU algorithm:         %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_AOS]));
    printf_m("SoA stack parallel GPU algorithm:         %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_SOA]));
}

const char * boolToEnabled(bool option)
{
    return option ? "enabled" : "disabled";
}

void parseGPUOptarg(const std::string & optarg)
{
    std::vector<std::string> substrings = { "all", "basic", "inc", "shm", "aos", "soa" };

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
            solversEnabled[PARALLEL_GPU_SHM] = true;
            break;
        case 4:
            solversEnabled[PARALLEL_GPU_AOS] = true;
            break;
        case 5:
            solversEnabled[PARALLEL_GPU_SOA] = true;
            break;
        default:
            printf("Error: Substring has no match!\n");
            break;
        }
    }

    gpuAlgorithmsToRun = std::count(solversEnabled.begin(), solversEnabled.end(), true);
}

const char * enumToKernelName(SolversEnableE solverType)
{
    const char* retVal = "";
    switch (solverType)
    {
    case PARALLEL_GPU_BASE:
        retVal = "parallelSolvingBase";
        break;
    case PARALLEL_GPU_INCREMENTAL:
        retVal = "parallelSolvingIncrementalStack";
        break;
    case PARALLEL_GPU_SHM:
        retVal = "parallelSolvingSharedMemory";
        break;
    case PARALLEL_GPU_AOS:
        retVal = "parallelSolvingAOSStack";
        break;
    case PARALLEL_GPU_SOA:
        retVal = "parallelSolvingSOAStack";
        break;
    default:
        break;
    }

    return retVal;
}

const char * enumToSolverName(SolversEnableE solverType)
{
    const char* retVal = "";
    switch (solverType)
    {
    case PARALLEL_GPU_BASE:
        retVal = "Base ParallelGpuSolver";
        break;
    case PARALLEL_GPU_INCREMENTAL:
        retVal = "Incremental ParallelGpuSolver";
        break;
    case PARALLEL_GPU_SHM:
        retVal = "Shared memory ParallelGpuSolver";
        break;
    case PARALLEL_GPU_AOS:
        retVal = "AOS stack ParallelGpuSolver";
        break;
    case PARALLEL_GPU_SOA:
        retVal = "SOA stack ParallelGpuSolver";
        break;
    default:
        break;
    }

    return retVal;
}

void getCurrentDirectory(TCHAR* buffer, size_t size)
{
#ifdef _WIN32
    GetCurrentDirectory(size, buffer);
#else
    if(getcwd(buffer, sizeof(*buffer) * size) == NULL)
    {
        perror("getcwd() error");
    }
#endif // _WIN32
}

void printPwd()
{
    TCHAR pwd[MAX_PWD_PATH] = {0};
    getCurrentDirectory(pwd, MAX_PWD_PATH);
    printf_m("Current directory: \"%s\"\n", pwd);
}

void printf_m(const char* format, ...)
{
    if (!muteOutput)
    {
        va_list args;
        va_start (args, format);
        vfprintf (stdout, format, args);
        va_end (args);
    }
}