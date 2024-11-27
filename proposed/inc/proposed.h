#ifndef PROPOSED_H
#define PROPOSED_H

#undef PRINT_INFO
// #define PRINT_INFO
#ifdef PRINT_INFO
    #define PDEBUG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define PDEBUG(fmt, ...)
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

