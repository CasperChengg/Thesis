#ifndef PROPOSED_H
#define PROPOSED_H

// #define DEBUG_MODE

#ifdef DEBUG_MODE
    #define PDEBUG(fmt, ...) printf(fmt, ##__VA_ARGS__)
    #define FDEBUG(fmt) fmt
#else
    #define PDEBUG(fmt, ...)
    #define FDEBUG(fmt)
#endif

#include <cmath>
#include <random>
#include <queue>
#include <utility> // std::pair
#include <iostream>
#include <algorithm>
#include "../../inc/validation.h"
#include "../../inc/file_operations.h"

void Proposed(std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const uint32_t k, const ModelParameters model_parameters);

#endif

