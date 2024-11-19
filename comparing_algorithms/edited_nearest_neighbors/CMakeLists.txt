cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../../inc
                    ${CMAKE_SOURCE_DIR}/inc
)

# Source files
set(ALL_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/../../src/decision_tree_classifier.cpp"
    "${CMAKE_SOURCE_DIR}/../../src/file_operations.cpp"
    "${CMAKE_SOURCE_DIR}/../../src/validation.cpp"
    "${CMAKE_SOURCE_DIR}/src/edited_nearest_neighbors.cpp"
    "${CMAKE_SOURCE_DIR}/src/main.cpp"
)

# Define configurable parameters with cache
set(KNN 5 CACHE STRING "Set number of NN")
set(TEST_TIME 20 CACHE STRING "Set test time")
set(K_FOLD 5 CACHE STRING "Set number of folds")
set(MODEL_TYPE "decision_tree" CACHE STRING "Set base classifier")
set(MIN_SAMPLES_SPLIT 10 CACHE STRING "Set minimum number of samples in a node to be split")
set(MAX_PURITY 0.95 CACHE STRING "Set maximum purity of nodes to be split")

# Add executable
add_executable(main ${ALL_SOURCE_FILES})

# Add compile definitions for main target
target_compile_definitions(main PRIVATE
    KNN=${KNN}
    TEST_TIME=${TEST_TIME}
    K_FOLD=${K_FOLD}
    MODEL_TYPE="${MODEL_TYPE}"
    MIN_SAMPLES_SPLIT=${MIN_SAMPLES_SPLIT}
    MAX_PURITY=${MAX_PURITY}
)
