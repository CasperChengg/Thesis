#include "../inc/edited_nearest_neighbors.h"

static uint32_t FindLeastMinorityClass(const std::vector<std::vector<float>> &training_set, 
                                                    const uint32_t n_classes)
{
    // Class labels start from 1, so n_classes requires (n_classes + 1) space for direct indexing
    std::vector<uint32_t> class_counts(n_classes + 1, 0); 
    const uint32_t label_idx = training_set[0].size() - 1;

    for(uint32_t data_idx = 0; data_idx < training_set.size(); data_idx++){   
        uint32_t label = training_set[data_idx][label_idx];
        class_counts[label]++;
    }

    uint32_t least_minority_class = 1;
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        if(class_counts[least_minority_class] > class_counts[class_idx]){
            least_minority_class = class_idx;
        }
    }
    
    return least_minority_class; 
}

static float EuclideanDistance(const std::vector<float> &src, const std::vector<float> &dst)
{
    float square_distance = 0;
    
    // last column stores label
    for(uint32_t feature_idx = 0; feature_idx < src.size() - 1; feature_idx++){
        float diff = src[feature_idx] - dst[feature_idx];
        square_distance += diff * diff;
    }
    return sqrt(square_distance);
}

static bool SameAsMajorityInKNN(const std::vector<std::vector<float>> &training_set, const uint32_t src_idx, const uint32_t k)
{
    uint32_t label_idx = training_set[0].size() - 1;
    uint32_t src_label = training_set[src_idx][label_idx];

    auto compare = [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b){return a.second < b.second;};
    std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, decltype(compare)> k_nearest_neighbors(compare);
    for(uint32_t dst_idx = 0; dst_idx < training_set.size(); dst_idx++){
        if(dst_idx != src_idx){
            k_nearest_neighbors.push({dst_idx, EuclideanDistance(training_set[src_idx], training_set[dst_idx])});
            if(k_nearest_neighbors.size() > k){
                k_nearest_neighbors.pop();
            }
        }
    }

    uint32_t n_same_label = 0;
    while(!k_nearest_neighbors.empty()){
        std::pair<uint32_t, float> nearest_neighbor = k_nearest_neighbors.top();
        uint32_t nearest_neighbor_idx   = nearest_neighbor.first;
        uint32_t nearest_neighbor_label = training_set[nearest_neighbor_idx][label_idx];
        if(nearest_neighbor_label == src_label){
            n_same_label++;
        }
        k_nearest_neighbors.pop();
    }
    
    return (float)n_same_label / k > 0.5;
}

void EditedNearestNeighbors(std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const uint32_t k)
{
    const uint32_t label_idx = training_set[0].size() - 1;
    const uint32_t least_minority_class = FindLeastMinorityClass(training_set, n_classes);
    std::vector<bool> is_reserved(training_set.size(), false);
    for(uint32_t data_idx = 0; data_idx < training_set.size(); data_idx++){
        if(training_set[data_idx][label_idx]  == least_minority_class || SameAsMajorityInKNN(training_set, data_idx, k)){
            is_reserved[data_idx] = true;
        }
    }

    for(int data_idx = training_set.size() - 1; data_idx >= 0; data_idx--){
        if(!is_reserved[data_idx]){
            training_set.erase(training_set.begin() + data_idx);
        }
    }
}