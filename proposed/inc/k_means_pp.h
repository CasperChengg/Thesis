#include <cmath>
#include <ctime>
#include <limits>
#include <vector>
#include <cstring>
#include <random>
#include <iostream>

template <class T>
unsigned int *KMeansPP(std::vector<std::vector<T>> &dataset, size_t n_clusters,  size_t max_iter, float tolerance);