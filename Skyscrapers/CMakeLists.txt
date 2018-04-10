# CMakeList.txt : CMake project for Skyscrapers, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project (Skyscrapers LANGUAGES CXX)

file(GLOB CPP_FILES *.cpp)
file(GLOB H_FILES *.h)

add_definitions(-DVERBOSITY_LEVEL=4)

# Add source to this project's executable.
add_library (${PROJECT_NAME} STATIC ${CPP_FILES} ${H_FILES})
add_dependencies(${PROJECT_NAME} Utilities)
target_link_libraries(${PROJECT_NAME} Utilities)
#add_dependencies(Skyscrapers Utilities)
# TODO: Add tests and install targets if needed.
