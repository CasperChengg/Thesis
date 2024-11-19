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

std::vector<uint32_t> FindKNearestNeighbors(const std::vector<std::vector<float>> &dataset, const uint32_t src_idx, const uint32_t k)
{
    std::vector<std::pair<uint32_t, float>> distance_pair(dataset.size(), std::pair<uint32_t, float>(0, 0.f));
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++)
    {
        distance_pair[data_idx].first  = data_idx;
        distance_pair[data_idx].second = EuclideanDistance(dataset[data_idx], dataset[src_idx]);
    }
    distance_pair[src_idx].second = std::numeric_limits<float>::max();
    sort(distance_pair.begin(),distance_pair.end(),[] (const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b ){return a.second < b.second;});

    std::vector<uint32_t> knn_idxes(k, 0);
    for(uint32_t nn_idx = 0; nn_idx < k; nn_idx++)
    {
        knn_idxes[nn_idx] = distance_pair[nn_idx].first;
    }
    
    return knn_idxes;
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

        std::vector<uint32_t> knn_idxes = FindKNearestNeighbors(training_set, training_data_idx, k);

        uint32_t num_same_label = 0;
        uint32_t training_data_label = training_set[training_data_idx][training_label_idx];
        for(uint32_t nn_idx = 0; nn_idx < k; nn_idx++){
            if(training_set[knn_idxes[nn_idx]][training_label_idx] == training_data_label){
                num_same_label++;
            }
        }

        if(num_same_label == k)
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