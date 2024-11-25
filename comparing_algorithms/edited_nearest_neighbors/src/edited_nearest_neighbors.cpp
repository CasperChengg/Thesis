#include "../inc/edited_nearest_neighbors.h"

uint32_t FindLeastMinorityClass(const std::vector<std::vector<float>> &training_set, 
                                                    const uint32_t n_classes)
{
    // Class labels start from 1, so n_classes requires (n_classes + 1) space for direct indexing
    std::vector<uint32_t> class_counts(n_classes + 1, 0); 
    const uint32_t label_idx = training_set[0].size() - 1;

    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){   
        uint32_t training_data_label = training_set[training_data_idx][label_idx];
        class_counts[training_data_label]++;
    }

    uint32_t least_minority_label = 1;
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        if(class_counts[least_minority_label] > class_counts[class_idx])
        {
            least_minority_label = class_idx;
        }
    }
    
    return least_minority_label; 
}

float EuclideanDistance(const std::vector<float> &src, const std::vector<float> &dst)
{
    float square_distance = 0;
    
    // last column stores label
    for(uint32_t i = 0; i < src.size() - 1; i++)
    {
        square_distance += pow(src[i] - dst[i], 2);
    }
    return sqrt(square_distance);
}

uint32_t CountNumSameLabelInKNN(const std::vector<std::vector<float>> &training_set, const uint32_t src_idx, const uint32_t k)
{
    uint32_t training_label_idx = training_set[0].size() - 1;
    uint32_t src_data_label = training_set[src_idx][training_label_idx];

    auto compare = [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b){return a.second < b.second;};
    std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, decltype(compare)> k_nearest_neighbors(compare);
    for(uint32_t dst_data_idx = 0; dst_data_idx < training_set.size(); dst_data_idx++){
        if(dst_data_idx == src_idx){
            continue;
        }
        k_nearest_neighbors.push({dst_data_idx, EuclideanDistance(training_set[src_idx], training_set[dst_data_idx])});
        if(k_nearest_neighbors.size() > k){
            k_nearest_neighbors.pop();
        }
    }

    uint32_t n_same_label = 0;
    while(!k_nearest_neighbors.empty()){
        std::pair<uint32_t, float> nearest_neighbor = k_nearest_neighbors.top();
        uint32_t nearest_neighbor_idx   = nearest_neighbor.first;
        uint32_t nearest_neighbor_label = training_set[nearest_neighbor_idx][training_label_idx];
        if(nearest_neighbor_label == src_data_label){
            n_same_label++;
        }

        k_nearest_neighbors.pop();
    }
    
    return n_same_label;
}

void EditedNearestNeighbors(std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const uint32_t k)
{
    const uint32_t training_label_idx = training_set[0].size() - 1;
    const uint32_t least_minority_label = FindLeastMinorityClass(training_set, n_classes);
    std::vector<bool> IsReserved(training_set.size(), false);

    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++)
    {
        if(training_set[training_data_idx][training_label_idx]  == least_minority_label){
            IsReserved[training_data_idx] = true;
            continue;
        }

        uint32_t n_same_label = CountNumSameLabelInKNN(training_set, training_data_idx, k);

        if(n_same_label == k)
        {    
            IsReserved[training_data_idx] = true;
        }
    }

    for(int training_data_idx = training_set.size() - 1; training_data_idx >= 0; training_data_idx--){
        if(!IsReserved[training_data_idx]){
            training_set.erase(training_set.begin() + training_data_idx);
        }
    }
}