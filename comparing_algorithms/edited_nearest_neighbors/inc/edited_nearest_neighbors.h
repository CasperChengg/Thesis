#ifndef EDIT_NEAREST_NEIGHBORS_H
#define EDIT_NEAREST_NEIGHBORS_H

#include <cmath>
#include <vector>
#include <cstdint>
#include <queue>
#include <utility> // std::pair
#include <algorithm>
#include <iostream>

void EditedNearestNeighbors(std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const uint32_t k);

#endif