#include "../inc/cluster_centroids.h"
static std::vector<uint32_t> CalculateClassCounts(const std::vector<std::vector<float>> &training_set, 
                                                    const uint32_t n_classes)
{
    std::vector<uint32_t> class_counts((n_classes + 1), 0); 

    const uint32_t training_label_idx = training_set[0].size() - 1;
    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){   
        uint32_t training_label = training_set[training_data_idx][training_label_idx];
        class_counts[training_label]++;
    }

    return class_counts;
}

void ClusterCentroids(std::vector<std::vector<float>> &training_set, 
                                                    const uint32_t n_classes, 
                                                        const uint32_t max_iters,  
                                                            const float tolerance)
{
    std::vector<uint32_t> class_counts = CalculateClassCounts(training_set, n_classes);
    
    std::vector<std::vector<float>> data_idxes_by_class[n_classes + 1];
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        data_idxes_by_class[class_idx].reserve(class_counts[class_idx]);
    }

    const uint32_t label_idx = training_set[0].size() - 1;
    for(uint32_t data_idx = 0; data_idx < training_set.size(); data_idx++){
        const uint32_t label = training_set[data_idx][label_idx];
        data_idxes_by_class[label].push_back(training_set[data_idx]);
    }
    training_set.clear();

    const uint32_t least_minority_sample_size = *std::min_element(class_counts.begin() + 1, class_counts.end());
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        std::vector<std::vector<float>> centroids = KMeansPP(data_idxes_by_class[class_idx], least_minority_sample_size, max_iters, tolerance);
        // KMeans centroids have no label
        for(uint32_t centroid_idx = 0; centroid_idx < centroids.size(); centroid_idx++){
            centroids[centroid_idx].push_back(class_idx);
        }
        training_set.insert(training_set.end(), centroids.begin(), centroids.end());
    }
}