# CMakeList.txt : CMake project for Skyscrapers, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project (Skyscrapers LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB CPP_FILES *.cpp)
file(GLOB H_FILES *.h)

add_definitions(-DVERBOSITY_LEVEL=4)

# Add source to this project's executable.
add_library (${PROJECT_NAME} STATIC ${CPP_FILES} ${H_FILES})
# Required on Unix OS family to be able to be linked into shared libraries.
set_target_properties(${PROJECT_NAME}
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
add_dependencies(${PROJECT_NAME} Utilities)
target_link_libraries(${PROJECT_NAME} Utilities)
target_include_directories (${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#add_dependencies(Skyscrapers Utilities)
# TODO: Add tests and install targets if needed.
