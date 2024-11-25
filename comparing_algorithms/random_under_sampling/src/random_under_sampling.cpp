#include "../inc/random_under_sampling.h"

static uint32_t CalculateSamplingSize(const std::vector<uint32_t> &class_counts)
{
    uint32_t smallest_class_count = std::numeric_limits<uint32_t>::max();
    for(uint32_t class_idx = 1; class_idx < class_counts.size(); class_idx++){
        if(smallest_class_count > class_counts[class_idx]){
            smallest_class_count = class_counts[class_idx];
        }
    }

    return smallest_class_count;
}

void RandomUnderSampling(std::vector<std::vector<float>> &training_set, const uint32_t n_classes)
{
    const uint32_t label_idx = training_set[0].size() - 1;
    std::vector<uint32_t> data_idxes_by_class[n_classes + 1];
    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){
        uint32_t training_data_label = training_set[training_data_idx][label_idx];
        data_idxes_by_class[training_data_label].push_back(training_data_idx);
    }

    std::vector<uint32_t> class_counts(n_classes + 1, 0);
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        class_counts[class_idx] = data_idxes_by_class[class_idx].size();
    }

    uint32_t sampling_size  = CalculateSamplingSize(class_counts);

    std::vector<std::vector<float>> preprocessed_data;
    preprocessed_data.reserve(n_classes * sampling_size); 
    for(uint32_t  class_idx = 1; class_idx <= n_classes; class_idx++){
        uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(data_idxes_by_class[class_idx].begin(), data_idxes_by_class[class_idx].end(), 
                                                                std::default_random_engine(seed));

        for(uint32_t shuffle_data_idx = 0; shuffle_data_idx < sampling_size; shuffle_data_idx++){
            uint32_t data_idx = data_idxes_by_class[class_idx][shuffle_data_idx];    
            preprocessed_data.push_back(training_set[data_idx]);
        }

    }
    training_set.clear();
    training_set = preprocessed_data;
}