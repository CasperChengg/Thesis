#include "edited_nearest_neighbors.h"

size_t FindLeastMinorityClass(Dataset *dataset)
{
    size_t minority_label = 1;
    for(size_t i = 2; i <= dataset->num_classes; i++)
    {
        if(dataset->training_set_class_counts[minority_label] > dataset->training_set_class_counts[i])
        {
            minority_label = i;
        }
    }
    return minority_label;
}

float EuclideanDistance(std::vector<float> &src, std::vector<float> &dst)
{
    float square_distance = 0;
    
    // last column stores label
    for(size_t i = 0; i < src.size() - 1; i++)
    {
        square_distance += pow(src[i] - dst[i], 2);
    }
    return sqrt(square_distance);
}

size_t* FindKNearestNeighbors(Dataset *dataset, std::vector<float> &src, size_t k)
{
    std::vector<std::pair<size_t, float>> distance_table;
    for(size_t i = 0; i < (dataset->training_set).size(); i++)
    {
        distance_table.push_back(std::make_pair(i, EuclideanDistance((dataset->training_set)[i], src)));
    }

    sort(distance_table.begin(),distance_table.end(),[] (const std::pair<size_t, float> &a, const std::pair<size_t, float> &b ){return a.second < b.second;});

    size_t *k_nearest_neighbors_index = (size_t*)malloc(k * sizeof(size_t));
    if(k_nearest_neighbors_index == NULL)
    {
        printf("./%s:%d: error: memory allocation error\n", __FILE__, __LINE__);
        exit(1);
    }
    memset(k_nearest_neighbors_index, 0, k * sizeof(size_t));

    for(size_t i = 0; i < k; i++)
    {
        k_nearest_neighbors_index[i] = distance_table[i + 1].first;
    }
    
    return k_nearest_neighbors_index;
}

void EditedNearestNeighbors(Dataset *dataset, size_t k){

    std::vector<std::vector<float>> preprocessed_data;
    size_t least_minority_label = FindLeastMinorityClass(dataset);
    for(size_t i = 0; i < (dataset->training_set).size(); i++)
    {
        if((dataset->training_set)[i][dataset->label_index]  == least_minority_label){
            preprocessed_data.push_back((dataset->training_set)[i]); 
            continue;
        }

        size_t *k_nearest_neighbors_index = FindKNearestNeighbors(dataset, (dataset->training_set)[i], k);

        size_t num_same_label = 0;
        for(size_t j = 0; j < k; j++)
        {
            if((dataset->training_set)[k_nearest_neighbors_index[j]][dataset->label_index] == (dataset->training_set)[i][dataset->label_index])
            {
                num_same_label++;
            }
        }

        if(num_same_label == k)
        {    
            preprocessed_data.push_back((dataset->training_set)[i]);
        }
        free(k_nearest_neighbors_index);
    }
    (dataset->training_set).clear();
    dataset->training_set = preprocessed_data;
    GetDatasetInfo(dataset, true);
}