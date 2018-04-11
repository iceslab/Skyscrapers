#include "../Utilities/asserts.h"
#include "../Utilities/Timer.h"
#include "../Skyscrapers/SequentialSolver.h"
#include "../Skyscrapers/ParallelCpuSolver.h"
#include <stdio.h>
#include "getopt.h"
#include "Statistics.h"

#include "KernelFunctions.inl"

#ifndef _WIN32
#include <unistd.h>
#define TCHAR char
#endif // !_WIN32

#define MAX_PWD_PATH 1024

bool loadFromFile;
const char* filePath;
std::vector<bool> solversEnabled(SOLVERS_SIZE, false);
size_t gpuAlgorithmsToRun;
size_t boardDimension = 1;
size_t threadsInBlock = 1;
size_t blocksOfThreads = 1;
size_t desiredBoards = 1;
size_t desiredFifoSize = CUDA_DEFAULT_FIFO_SIZE;

Statistics launchSequentialSolver(const board::Board & board);
Statistics launchParallelCpuSolver(const board::Board & board);
Statistics launchGenericGpuSolver(const board::Board board, SolversEnableE solverType);

// Command line processing functions
bool ProcessCommandLine(int argc, char *argv[]);
void printUsage();
void printLaunchParameters();
const char *boolToEnabled(bool option);
void parseGPUOptarg(const std::string &optarg);
const char *enumToKernelName(SolversEnableE solverType);
const char *enumToSolverName(SolversEnableE solverType);
void getCurrentDirectory(TCHAR *buffer, size_t size);

int main(int argc, char** argv)
{
    TCHAR pwd[MAX_PWD_PATH];
    getCurrentDirectory(pwd, MAX_PWD_PATH);
    std::cout << "Current directory: \"" << pwd << "\"" << std::endl;
    if (ProcessCommandLine(argc, argv) == false)
    {
        // Exit when commandline processing fails
        printUsage();
        return -1;
    }

    printLaunchParameters();
    // Prepare data on host
    board::Board b(1);
    if (loadFromFile == true)
    {
        if (!b.readFromFile(filePath))
        {
            printf("Could not read file \"%s\"", filePath);
            return -2;
        }
        b.calculateHints();
    }
    else
    {
        b.generate(boardDimension);
    }
    b.saveToFile("lastRun.txt");

    printf("\nExpected result:\n");
    printf("==========================\n");
    b.print();
    printf("==========================\n");
    fflush(stdout);
    fflush(stderr);

    // CPU solvers
    const auto sStats = launchSequentialSolver(b);
    const auto pcStats = launchParallelCpuSolver(b);
    const auto bpgStats = launchGenericGpuSolver(b, PARALLEL_GPU_BASE);
    const auto isgStats = launchGenericGpuSolver(b, PARALLEL_GPU_INCREMENTAL);
    const auto shmgStats = launchGenericGpuSolver(b, PARALLEL_GPU_SHM);
    const auto aosgStats = launchGenericGpuSolver(b, PARALLEL_GPU_AOS);
    const auto soagStats = launchGenericGpuSolver(b, PARALLEL_GPU_SOA);

    sStats.print();
    pcStats.print();
    bpgStats.print();
    isgStats.print();
    shmgStats.print();
    aosgStats.print();
    soagStats.print();

    //system("pause");
    return 0;
}

Statistics launchSequentialSolver(const board::Board & board)
{
    Statistics retVal(solversEnabled[SEQUENTIAL]);
    auto cMilliseconds = std::numeric_limits<double>::quiet_NaN();
    if (solversEnabled[SEQUENTIAL] == true)
    {
        solver::SequentialSolver c(board);
        Timer time;
        time.start();
        const auto cResult = c.solve();
        cMilliseconds = time.stop(Resolution::MILLISECONDS);

        auto validStats = board::Board::validateResults(cResult);
        std::cout
            << "SequentialSolver result size: "
            << validStats.allBoards
            << ", valid solutions: "
            << validStats.validSolutions
            << ", repeated solutions: "
            << validStats.repeatedSolutions
            << std::endl;
    }
    retVal.emplace_back("+ SequentialSolver solving time: ", cMilliseconds);
    return retVal;
}

Statistics launchParallelCpuSolver(const board::Board & board)
{
    Statistics retVal(solversEnabled[PARALLEL_CPU]);
    auto pcMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto initMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto generationMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto threadsLaunchMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto threadsSyncMilliseconds = std::numeric_limits<double>::quiet_NaN();
    if (solversEnabled[PARALLEL_CPU] == true)
    {
        solver::ParallelCpuSolver pc(board);
        Timer time;
        time.start();
        const auto pcResult = pc.solve(desiredBoards,
                                       initMilliseconds,
                                       generationMilliseconds,
                                       threadsLaunchMilliseconds,
                                       threadsSyncMilliseconds);
        pcMilliseconds = time.stop(Resolution::MILLISECONDS);
        auto validStats = board::Board::validateResults(pcResult);

        for (size_t i = 0; i < pcResult.size(); i++)
        {
            pcResult[i].saveToFile("res_board_" + std::to_string(i) + ".txt");
        }

        std::cout
            << "ParallelCpuSolver result size: "
            << validStats.allBoards
            << ", valid solutions: "
            << validStats.validSolutions
            << ", repeated solutions: "
            << validStats.repeatedSolutions
            << std::endl;
    }
    retVal.emplace_back("+ ParallelCpuSolver solving time: ", pcMilliseconds);
    retVal.emplace_back("\nDevice initialize time: ", initMilliseconds);
    retVal.emplace_back("Board generation time: ", generationMilliseconds);
    retVal.emplace_back("Threads launch time: ", threadsLaunchMilliseconds);
    retVal.emplace_back("Threads synchronize time: ", threadsSyncMilliseconds);

    return retVal;
}

Statistics launchGenericGpuSolver(const board::Board board, SolversEnableE solverType)
{
    Statistics retVal(false);
    if (solverType < PARALLEL_GPU_BEGIN || solverType > PARALLEL_GPU_END)
    {
        printf("Unsupported GPU algorithm\n");
        return retVal;
    }
    retVal.printable = solversEnabled[solverType];

    auto pgMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto initMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto deinitMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto generationMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto allocationMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto deallocationMilliseconds = std::numeric_limits<double>::quiet_NaN();

    auto kernelLaunchMilliseconds = std::numeric_limits<double>::quiet_NaN();
    auto kernelSyncMilliseconds = std::numeric_limits<double>::quiet_NaN();

    if (solversEnabled[solverType] == true)
    {
        Timer time;
        Timer timeInit;
        Timer timeGeneration;
        Timer timeAllocation;
        time.start();
        // Initialize device
        cuda::initDevice();
        initMilliseconds = time.stop(Resolution::MILLISECONDS);

        solver::ParallelSolver ps(board);
        timeGeneration.start();
        const auto boards = ps.generateBoards(desiredBoards);
        generationMilliseconds = timeGeneration.stop(Resolution::MILLISECONDS);
        size_t generatedSolversCount = 0;

        timeAllocation.start();
        // Host vector for solvers - needed to properly execute destructors
        // It's lifetime ensures that pointers on device are valid during kernel execution
        std::vector<cuda::solver::SequentialSolver> h_solvers;
        // Host vector for boards - needed to properly execute destructors
        // It's lifetime ensures that pointers on device are valid during kernel execution
        std::vector<cuda::Board> h_boards;

        // Allocating memory on device
        auto d_solvers = cuda::solver::prepareSolvers(boards, h_solvers, generatedSolversCount);
        auto d_outputBoards = cuda::solver::prepareResultArray(h_boards, generatedSolversCount, board.size());
        auto d_outputBoardsSize = cuda::solver::prepareResultArraySize();
        auto d_solversTaken = cuda::solver::prepareSolversTaken();
        auto d_generatedSolversCount = cuda::solver::prepareGeneratedSolversCount(generatedSolversCount);
        printf("Generated %zu solvers\n", generatedSolversCount);

        std::vector<cuda::cudaEventsDeviceT> h_timers(generatedSolversCount);
        auto d_timers = cuda::solver::prepareCudaEventDevice(h_timers);
        auto d_stack = cuda::solver::prepareStack(solverType, generatedSolversCount, board.getCellsCount());
        cuda::solver::prepareConstantMemory(board);

        // Allocating memory on host
        auto h_outputBoards = cuda::solver::prepareHostResultArray();
        cuda::uint32T h_outputBoardsSize = 0;
        allocationMilliseconds = timeAllocation.stop(Resolution::MILLISECONDS);

        // If allocation was successfull launch kernel
        if (cuda::solver::verifyAllocation(d_solvers, d_outputBoards, d_outputBoardsSize))
        {
            dim3 numBlocks(blocksOfThreads);
            dim3 threadsPerBlock(threadsInBlock);
            int sharedMemorySize = cuda::solver::getSharedMemorySize(solverType,
                                                                     generatedSolversCount,
                                                                     board.getCellsCount());
            Timer kernelTimer;
            if (sharedMemorySize <= CUDA_MAX_SHARED_MEMORY)
            {
                printf("Launching kernel...\n");
                fflush(stdout);
                fflush(stderr);

                kernelTimer.start();
                launchGenericGpuKernel(solverType,
                                       numBlocks,
                                       threadsPerBlock,
                                       sharedMemorySize,
                                       d_solvers,
                                       d_solversTaken,
                                       d_generatedSolversCount,
                                       d_outputBoards,
                                       d_outputBoardsSize,
                                       d_stack,
                                       d_timers);
                kernelLaunchMilliseconds = kernelTimer.stop(Resolution::MILLISECONDS);
            }
            else
            {
                auto shmConverted = cuda::bytesToHumanReadable(sharedMemorySize);
                auto limitConverted = cuda::bytesToHumanReadable(CUDA_MAX_SHARED_MEMORY);
                printf("Exceeded shared memory size limit (%5.1f %s/%5.1f %s)."
                       "Kernel launch aborted\n",
                       shmConverted.first,
                       shmConverted.second.c_str(),
                       limitConverted.first,
                       limitConverted.second.c_str());
            }

            // Check for any errors launching the kernel
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                printf("%s launch failed: %s\n",
                       enumToKernelName(solverType),
                       cudaGetErrorString(cudaStatus));
            }
            else
            {
                // cudaDeviceSynchronize waits for the kernel to finish, and returns
                // any errors encountered during the launch.
                cudaStatus = cudaDeviceSynchronize();
                kernelSyncMilliseconds = kernelTimer.stop(Resolution::MILLISECONDS);
                printf("Kernel finished\n");
                fflush(stdout);
                fflush(stderr);
                if (cudaStatus != cudaSuccess)
                {
                    printf("cudaDeviceSynchronize returned %d \"%s\"\n",
                           cudaStatus,
                           cudaGetErrorString(cudaStatus));
                }
                else
                {
                    cuda::solver::copyCudaEventDevice(h_timers, d_timers);
                    for (size_t i = 0; i < h_timers.size(); i++)
                    {
                        const auto & el = h_timers[i];
                        double initTime = cuda::getTime(el.initBegin,
                                                        el.initEnd,
                                                        cuda::Resolution::MILLISECONDS);
                        double loopTime = cuda::getTime(el.loopBegin,
                                                        el.loopEnd,
                                                        cuda::Resolution::MILLISECONDS);
                        double firstZeroDiff = cuda::getTime(el.firstZeroDiff,
                                                             cuda::Resolution::MILLISECONDS);
                        double goodIndexDiff = cuda::getTime(el.goodIndexDiff,
                                                             cuda::Resolution::MILLISECONDS);
                        double badIndexDiff = cuda::getTime(el.badIndexDiff,
                                                            cuda::Resolution::MILLISECONDS);
                        double placeableDiff = cuda::getTime(el.placeableDiff,
                                                             cuda::Resolution::MILLISECONDS);
                        double placeableFnDiff = cuda::getTime(el.placeableFnDiff,
                                                               cuda::Resolution::MILLISECONDS);
                        double boardValidDiff = cuda::getTime(el.boardValidDiff,
                                                              cuda::Resolution::MILLISECONDS);
                        double boardInvalidDiff = cuda::getTime(el.boardInvalidDiff,
                                                                cuda::Resolution::MILLISECONDS);
                        double boardValidFnDiff = cuda::getTime(el.boardValidFnDiff,
                                                                cuda::Resolution::MILLISECONDS);
                        double lastCellDiff = cuda::getTime(el.lastCellDiff,
                                                            cuda::Resolution::MILLISECONDS);
                        double notLastCellDiff = cuda::getTime(el.notLastCellDiff,
                                                               cuda::Resolution::MILLISECONDS);
                        double copyResultDiff = cuda::getTime(el.copyResultDiff,
                                                              cuda::Resolution::MILLISECONDS);

                        //retVal.emplace_back("\nThread #" + std::to_string(i) + " init time: ", initTime);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " loop time: ", loopTime);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " first zero time: ", firstZeroDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " good index time: ", goodIndexDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " bad index time: ", badIndexDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " is placeable time: ", placeableDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " is placeable function time: ", placeableFnDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " board valid time: ", boardValidDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " board invalid time: ", boardInvalidDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " board valid function time: ", boardValidFnDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " last cell time: ", lastCellDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " not last cell time: ", notLastCellDiff);
                        //retVal.emplace_back("Thread #" + std::to_string(i) + " copy result time: ", copyResultDiff);
                    }

                    cuda::solver::copyResultsArray(h_outputBoards,
                                                   d_outputBoards,
                                                   generatedSolversCount);
                    cuda::solver::copyResultsArraySize(&h_outputBoardsSize, d_outputBoardsSize);

                    std::vector<board::Board> h_resultBoards;
                    h_resultBoards.reserve(h_outputBoardsSize);

                    for (size_t i = 0; i <= h_outputBoardsSize && i < CUDA_MAX_RESULTS; i++)
                    {
                        board::Board b(h_outputBoards[i].getHostVector());
                        b.calculateHints();
                        h_resultBoards.emplace_back(b);
                        //b.print();
                    }

                    auto validStats = board::Board::validateResults(h_resultBoards);

                    std::cout
                        << enumToSolverName(solverType)
                        << " result size: "
                        << validStats.allBoards
                        << ", valid solutions: "
                        << validStats.validSolutions
                        << ", repeated solutions: "
                        << validStats.repeatedSolutions
                        << std::endl;
                }
            }
        }

        timeAllocation.start();
        // Dellocating host memory (in reverse order)
        cuda::solver::freeHostResultArray(h_outputBoards);

        // Dellocating device memory (in reverse order)
        cuda::solver::freeStack(d_stack);
        cuda::solver::freeCudaEventDevice(d_timers);
        cuda::solver::freeGeneratedSolversCount(d_generatedSolversCount);
        cuda::solver::freeSolversTaken(d_solversTaken);
        cuda::solver::freeResultArraySize(d_outputBoardsSize);
        cuda::solver::freeResultArray(d_outputBoards);
        cuda::solver::freeSolvers(d_solvers);
        deallocationMilliseconds = timeAllocation.stop(Resolution::MILLISECONDS);

        timeInit.start();
        // Deinitialize device
        cuda::deinitDevice();
        deinitMilliseconds = timeInit.stop(Resolution::MILLISECONDS);
        pgMilliseconds = time.stop(Resolution::MILLISECONDS);
    }

    std::stringstream ss;
    ss << "+ " << enumToSolverName(solverType) << " solving time: ";

    retVal.emplace_back(ss.str(), pgMilliseconds);
    retVal.emplace_back("\nDevice initialize time: ", initMilliseconds);
    retVal.emplace_back("Device deinitialize time: ", deinitMilliseconds);
    retVal.emplace_back("\nBoard generation time: ", generationMilliseconds);
    retVal.emplace_back("\nMemory allocation time: ", allocationMilliseconds);
    retVal.emplace_back("Memory deallocation time: ", deallocationMilliseconds);
    retVal.emplace_back("\nKernel launch time: ", kernelLaunchMilliseconds);
    retVal.emplace_back("Kernel synchronize time: ", kernelSyncMilliseconds);
    retVal.emplace_back("Allocation + synchronize + deallocation time: ",
                        allocationMilliseconds + kernelSyncMilliseconds + deallocationMilliseconds);

    return retVal;
}

bool ProcessCommandLine(int argc, char *argv[])
{
    int c;

    while ((c = getopt(argc, argv, "f:scg:d:t:b:p:q:h")) != EOF)
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
        case 'h':
            return false;
            break;
        case '?':
            printf("ERROR: illegal option %s\n", argv[optind - 1]);
            return false;
            break;
        default:
            printUsage();
            printf("WARNING: no handler for option %c\n", c);
            return false;
            break;
        }
    }

    //
    // check for non-option args here
    //
    return true;
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

    printf("Number of threads in parallel CPU algorithms: %s\n", "equal to generated boards");
    printf("Number of threads in parallel GPU algorithms: %7zu\n", threadsInBlock * blocksOfThreads);
    printf("Number of threads per block:                  %7zu\n", threadsInBlock);
    printf("Number of blocks of threads:                  %7zu\n", blocksOfThreads);
    printf("Sequential algorithm:                     %s\n", boolToEnabled(solversEnabled[SEQUENTIAL]));
    printf("Parallel CPU algorithm:                   %s\n", boolToEnabled(solversEnabled[PARALLEL_CPU]));
    printf("Base parallel GPU algorithm:              %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_BASE]));
    printf("Incremental stack parallel GPU algorithm: %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_INCREMENTAL]));
    printf("Shared memory parallel GPU algorithm:     %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_SHM]));
    printf("AoS stack parallel GPU algorithm:         %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_AOS]));
    printf("SoA stack parallel GPU algorithm:         %s\n", boolToEnabled(solversEnabled[PARALLEL_GPU_SOA]));
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