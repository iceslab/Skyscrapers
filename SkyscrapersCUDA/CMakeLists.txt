# CMakeList.txt : CMake project for SkyscrapersCUDA, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project (SkyscrapersCUDA LANGUAGES CXX CUDA)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

file(GLOB CPP_FILES *.cpp)
file(GLOB C_FILES *.c)
file(GLOB H_FILES *.h)
file(GLOB CU_FILES *.cu)
file(GLOB CUH_FILES *.cuh)
file(GLOB KERNEL_FILE kernel.cu)
set(PROJECT_LIB_NAME "${PROJECT_NAME}_Static")

list(FILTER CU_FILES EXCLUDE REGEX ".*kernel.cu$")

# Remove MinGW when not on Windows
if(WIN32)
    # Nothing to do
else()
    list(FILTER C_FILES EXCLUDE REGEX ".*getopt.c$")
    list(FILTER H_FILES EXCLUDE REGEX ".*getopt.h$")
endif()

add_definitions(-DVERBOSITY_LEVEL=4)
set(CUDA_64_BIT_DEVICE_CODE ON)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(${PROJECT_LIB_NAME} ${CPP_FILES} ${C_FILES} ${H_FILES} ${CU_FILES} ${CUH_FILES})
set_target_properties(${PROJECT_LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                                     CXX_STANDARD 11
                                                     CUDA_STANDARD 11)
add_executable(${PROJECT_NAME} ${KERNEL_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                                 CXX_STANDARD 11
                                                 CUDA_STANDARD 11)
target_link_libraries(${PROJECT_NAME} Utilities Skyscrapers ${PROJECT_LIB_NAME})
