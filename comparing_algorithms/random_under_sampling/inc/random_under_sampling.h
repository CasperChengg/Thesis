#ifndef RANDOM_UNDER_SAMPLING_H
#define RANDOM_UNDER_SAMPLING_H

#include <random>    // std::default_random_engine
#include <chrono>    // std::chrono  
#include <algorithm> // shuffle
#include <iostream>
#include "./file_operations.h"

void RandomUnderSampling(std::vector<std::vector<float>> &training_set, const uint32_t n_classes);

#endif
