cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(inc)

# Source files
file(GLOB_RECURSE ALL_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/inc/*.h"
)

set(TEST_TIME 20 CACHE STRING "Set test time")
set(K_FOLD 5 CACHE STRING "Set number of folds")

set(MODEL_TYPE "decision_tree" CACHE STRING "Set base classifier")
set(MIN_SAMPLES_SPLIT 10 CACHE STRING "Set minimum number of samples in a nodes to be split")
set(MAX_PURITY 0.95 CACHE STRING "Set maximum purity of nodes to be split")

add_definitions(-DTEST_TIME=${TEST_TIME})
add_definitions(-DK_FOLD=${K_FOLD})
add_definitions(-DMODEL_TYPE="${MODEL_TYPE}")
add_definitions(-DMIN_SAMPLES_SPLIT=${MIN_SAMPLES_SPLIT})
add_definitions(-DMAX_PURITY=${MAX_PURITY})

# Add executable
add_executable(main ${ALL_SOURCE_FILES})