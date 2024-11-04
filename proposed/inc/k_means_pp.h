// #define DEBUG
#include <cmath> // sqrt
#include <vector>
#include <random> // std::random_device, std::mt19937 gen(), std::uniform_real_distribution<>;
#include <limits> // std::numeric_limits<float>::max();
#ifdef DEBUG
#include<iostream>
#endif

std::vector<std::vector<float>> KMeansPP(std::vector<std::vector<float>> &dataset, uint32_t n_clusters,  uint32_t max_iter, float tolerance);
