#include "./random_under_sampling.h"

size_t FindLeastMinorityClass(Dataset *dataset)
{
    size_t minority_label = 1;
    for(size_t i = 2; i <= dataset->num_classes; i++)
    {
        if((dataset->training_set_class_counts)[minority_label] > (dataset->training_set_class_counts)[i])
        {
            minority_label = i;
        }
    }
    return minority_label;
}

void RandomUnderSampling(Dataset *dataset)
{
    size_t least_minority_label = FindLeastMinorityClass(dataset);
    size_t sampling_size  = (dataset->training_set_class_counts)[least_minority_label];

    std::vector<size_t> sample_index_in_class[dataset->num_classes + 1];
    for(size_t i = 0; i < (dataset->training_set).size(); i++)
    {
        size_t label = (dataset->training_set)[i][dataset->label_index];
        sample_index_in_class[label].push_back(i);
    }

    std::vector<std::vector<float>> preprocessed_data;
    for(size_t i = 1; i <= dataset->num_classes; i++)
    {
        if(i != least_minority_label)
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            shuffle(sample_index_in_class[i].begin(), sample_index_in_class[i].end(), std::default_random_engine(seed));
        }

        for(size_t j = 0; j < sampling_size; j++)
        {
            preprocessed_data.push_back(dataset->training_set[sample_index_in_class[i][j]]);
        }
    }
    dataset->training_set.clear();
    dataset->training_set = preprocessed_data; 
    GetDatasetInfo(dataset, true);
}