cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA support
# enable_language(CUDA)

# Set CUDA standard
# set(CMAKE_CUDA_STANDARD 11)
# set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Include directories
include_directories(inc)

# Source files
file(GLOB_RECURSE ALL_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/inc/*.h"
)

set(KNN 5 CACHE STRING "Set number of NN")
set(TEST_TIME 20 CACHE STRING "Set test time")
set(K_FOLD 5 CACHE STRING "Set number of folds")
set(ETA 10 CACHE STRING "Set largest number of nodes in a decision tree")
set(PI 0.95 CACHE STRING "Set smallest purity of nodes in a decision tree")

add_definitions(-DKNN=${KNN})
add_definitions(-DTEST_TIME=${TEST_TIME})
add_definitions(-DK_FOLD=${K_FOLD})
add_definitions(-DETA=${ETA})
add_definitions(-DPI=${PI})

# 定义可以从命令行启用的选项
option(USE_KNN "Define USE_KNN" OFF)
option(CONSIDER_MAJ "Define CONSIDER_MAJ" OFF)
option(CONSIDER_MIN "Define CONSIDER_MIN" OFF)
option(KNN_DISTANCE "Define KNN_DISTANCE" OFF)
option(DIVIDE_BY_DISTANCE "Define DIVIDE_BY_DISTANCE" OFF)
option(MULTIPLE_BY_DISTANCE "Define MULTIPLE_BY_DISTANCE" OFF)

option(USE_RNN "Define USE_RNN" OFF)

# 根据传入的选项添加宏定义
if(USE_KNN)
    add_definitions(-DUSE_KNN)
endif()

if(CONSIDER_MAJ)
    add_definitions(-DCONSIDER_MAJ)
endif()

if(CONSIDER_MIN)
    add_definitions(-DCONSIDER_MIN)
endif()

if(KNN_DISTANCE)
    add_definitions(-DKNN_DISTANCE)
endif()

if(DIVIDE_BY_DISTANCE)
    add_definitions(-DDIVIDE_BY_DISTANCE)
endif()

if(MULTIPLE_BY_DISTANCE)
    add_definitions(-DMULTIPLE_BY_DISTANCE)
endif()

if(USE_RNN)
    add_definitions(-DUSE_RNN)
endif()


# add_definitions(-DRNN)


# Add executable
add_executable(main ${ALL_SOURCE_FILES})

# Add CUDA library
# find_package(CUDA REQUIRED)
# target_link_libraries(MyProject ${CUDA_LIBRARIES})