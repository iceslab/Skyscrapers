#include "asserts.h"
#include "Timer.h"
#include "../Skyscrappers/SequentialSolver.h"
#include "../Skyscrappers/ParallelCpuSolver.h"
#include <stdio.h>
#include "XGetopt.h"
#include "Statistics.h"

#include "KernelFunctions.inl"

Statistics launchSequentialSolver(const board::Board & board);
Statistics launchParallelCpuSolver(const board::Board & board);

// TODO: Make generic GPU launch function
Statistics launchGenericGpuSolver(const board::Board & board, SolversEnableE solverType);

int main(int argc, char** argv)
{
    TCHAR pwd[MAX_PATH];
    GetCurrentDirectory(MAX_PATH, pwd);
    std::cout << "Current directory: \"" << pwd << "\"" << std::endl;
    if (ProcessCommandLine(argc, argv) == FALSE)
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
        std::cout << "SequentialSolver result size: " << cResult.size() << std::endl;
    }
    retVal.emplace_back("SequentialSolver solving time: ", cMilliseconds);
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
        std::cout << "ParallelCpuSolver result size: " << pcResult.size() << std::endl;
    }
    retVal.emplace_back("ParallelCpuSolver solving time: ", pcMilliseconds);
    retVal.emplace_back("\nDevice initialize time: ", initMilliseconds);
    retVal.emplace_back("Board generation time: ", generationMilliseconds);
    retVal.emplace_back("Threads launch time: ", threadsLaunchMilliseconds);
    retVal.emplace_back("Threads synchronize time: ", threadsSyncMilliseconds);

    return retVal;
}

Statistics launchGenericGpuSolver(const board::Board & board, SolversEnableE solverType)
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
            int sharedMemorySize = cuda::solver::getSharedMemorySize(solverType);

            printf("Launching kernel...\n");
            fflush(stdout);
            fflush(stderr);

            Timer kernelTimer;
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
                    for (size_t i = 0; i <= h_outputBoardsSize && i < CUDA_MAX_RESULTS; i++)
                    {
                        board::Board b(h_outputBoards[i].getHostVector());
                        b.calculateHints();
                        //b.print();
                    }

                    std::cout
                        << enumToSolverName(solverType)
                        << " result size: "
                        << h_outputBoardsSize
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
