#ifndef PROPOSED_H
#define PROPOSED_H

#include <cmath>
#include <bitset>
#include <utility>
#include <iostream>
#include <algorithm>
#include "k_means_pp.h"
#include "prim.h"
#include "file_operations.h"

typedef struct fitness
{
    uint32_t minority_rnn_counts;
    float distance_to_minority_rnn;
    float fitness;
}fitness;

template <typename T>
void Proposed(std::vector<std::vector<T>>dataset, uint32_t n_classes, uint32_t k);

#endif