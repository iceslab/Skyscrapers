# CMakeList.txt : CMake project for SkyscrapersCUDA, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project (SkyscrapersCUDA LANGUAGES CXX CUDA)

#Searching CUDA
#set(CUDA_SEPARABLE_COMPILATION ON)
#find_package(CUDA REQUIRED)
#get_filename_component(CUDA_LIB_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)
#find_library(CUDA_cudadevrt_LIBRARY cudadevrt PATHS ${CUDA_LIB_PATH})
#find_library(CUDA_cudart_LIBRARY cudart PATHS ${CUDA_LIB_PATH})

file(GLOB CPP_FILES *.cpp)
file(GLOB H_FILES *.h)
file(GLOB CU_FILES *.cu)
file(GLOB CUH_FILES *.cuh)
file(GLOB KERNEL_FILE kernel.cu)
#file(GLOB CUDA_LIBRARIES "${CUDA_LIB_PATH}/*")
set(PROJECT_LIB_NAME "${PROJECT_NAME}_Static")

#list(REMOVE_ITEM CU_FILES "kernel.cu")
list(FILTER CU_FILES EXCLUDE REGEX ".*kernel.cu$")
message("CU_FILES: ${CU_FILES}")
message("KERNEL_FILE: ${KERNEL_FILE}")

add_definitions(-DVERBOSITY_LEVEL=4)
 
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -rdc=true")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

set(CUDA_64_BIT_DEVICE_CODE ON)

message("CUDA_64_BIT_DEVICE_CODE: ${CUDA_64_BIT_DEVICE_CODE}")

add_library(${PROJECT_LIB_NAME} ${CPP_FILES} ${H_FILES} ${CU_FILES} ${CUH_FILES})
set_target_properties(${PROJECT_LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Add source to this project's executable.
#cuda_add_executable (${PROJECT_NAME} ${CPP_FILES} ${H_FILES} ${CU_FILES} ${CUH_FILES})
#add_executable(${PROJECT_NAME} ${CPP_FILES} ${H_FILES} ${CU_FILES} ${CUH_FILES})
add_executable(${PROJECT_NAME} ${KERNEL_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#message("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
#message("CUDA_cudadevrt_LIBRARY: ${CUDA_cudadevrt_LIBRARY}")
#message("CUDA_cudart_LIBRARY: ${CUDA_cudart_LIBRARY}")
#message("CUDA_LIB_PATH: ${CUDA_LIB_PATH}")

#link_directories(${CUDA_LIB_PATH})

#add_dependencies(${PROJECT_NAME} Utilities Skyscrapers SkyscrapersCUDAStatic)
#add_dependencies(${PROJECT_NAME} Utilities Skyscrapers)
target_link_libraries(${PROJECT_NAME} Utilities Skyscrapers ${PROJECT_LIB_NAME})
#target_link_libraries(${PROJECT_NAME} Utilities Skyscrapers ${CUDA_LIBRARIES})
# TODO: Add tests and install targets if needed.
