#include<vector>
#include<algorithm>
#include<cstdint>
#include<iostream>
#include "./k_means_pp.h"

void ClusterCentroids(std::vector<std::vector<float>> &training_set, 
                        const uint32_t n_classes, 
                            const uint32_t max_iters,  
                                const float tolerance);