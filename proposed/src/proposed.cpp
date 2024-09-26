#include "./proposed.h"

bool* FindMinority(Dataset *dataset)
{
    bool* is_minority = (bool*)malloc((dataset->num_classes + 1) * sizeof(bool));
    if(is_minority == NULL)
    {
        fprintf(stderr, "./%s:%d: \033[31merror\033[0m: memory allocation error\n", __FILE__, __LINE__);
        exit(1);
    }
    for(size_t i = 0; i < dataset->num_classes + 1; i++) is_minority[i] = false;

    float avg_class_counts = 0;
    for(size_t i = 1; i <= dataset->num_classes; i++)
    {
        avg_class_counts += (float)dataset->training_set_class_counts[i];
    }
    avg_class_counts /= dataset->num_classes;;

    for(size_t i = 1; i <= dataset->num_classes; i++)
    {
        if(dataset->training_set_class_counts[i] <= avg_class_counts)
        {
            is_minority[i] = true;
        }
    }
    return is_minority;
}

float EuclideanDistance(std::vector<float> &src, std::vector<float> &dst)
{
    float square_distance = 0;
    // the last column of vector stores the label
    for(size_t i = 0; i < src.size() - 1; i++)
    {
        square_distance += pow(src[i] - dst[i], 2);
    }
    return sqrt(square_distance);
}

std::vector<std::vector<float>> FindKNearestNeighbors(Dataset *dataset, std::vector<float> &src, size_t k)
{
    std::vector<std::vector<float>> distance_table;
    std::vector<float> index_with_distance;
    for(size_t i = 0; i < (dataset->training_set).size(); i++)
    {
        index_with_distance.push_back(i);
        index_with_distance.push_back(EuclideanDistance((dataset->training_set)[i], src));
        distance_table.push_back(index_with_distance);
        index_with_distance.clear();
    }
    
    sort(distance_table.begin(),distance_table.end(),[] (const std::vector<float> &a, const std::vector<float> &b ){return a[1] < b[1];});

    std::vector<std::vector<float>> k_nearest_neighbors;
    for(int i = 1; i <= k; i++)
    {
        k_nearest_neighbors.push_back(distance_table[i]);
    }
    
    return k_nearest_neighbors;
}

size_t BinarySearchRange(std::vector<float> array, float target)
{
    size_t start = 0, end = array.size() - 1;
    if(target < array[start])
    {
        return start;
    }
    else if(target >= array[end])
    {
        return end;
    }
    
    while((end- start) > 1)
    {
        size_t middle = start + (end - start) / 2;
        if(target > array[middle])
        {
            start = middle;
        }
        else if(target < array[middle])
        {
            end = middle;
        }
        else
        {
            return middle;
        }
        
    }
    return end;
}

std::vector<size_t> RouletteWheelSelection(std::vector<float> fitness, size_t selection_size)
{
    float total_fitness = 0.0;
    size_t num_nonzero = 0;

    if(fitness.size() < selection_size)
    {
        std::vector<size_t> selected_index;
        for(size_t i = 0; i < fitness.size(); i++){
            selected_index.push_back(i);
        }
        return selected_index;
    }

    for(size_t i = 0; i < fitness.size(); i++)
    {
        if(fitness[i] > 0)
        {
            num_nonzero++;
            total_fitness += fitness[i];
        }
    }

    if(num_nonzero < selection_size)
    {
        total_fitness = 0;
        for(size_t i = 0; i < fitness.size(); i++)
        {
            if(fitness[i] == 0)
            {
                fitness[i] = 1e-2;
            }
            total_fitness += fitness[i];
        } 
    }

    fitness[0] /= total_fitness;
    for(size_t pocket = 1; pocket < fitness.size(); pocket++)
    {
        fitness[pocket] = fitness[pocket - 1] + fitness[pocket] / total_fitness;
    }

    srand(time(NULL));
    bool *is_selected = (bool*)malloc(fitness.size() * sizeof(bool));
    for(size_t i = 0; i < fitness.size(); i++) is_selected[i] = false;

    for(size_t i = 0; i < selection_size; i++)
    {
        float hit_point = (float)rand() / INT32_MAX;

        size_t pocket = BinarySearchRange(fitness, hit_point);
        if(is_selected[pocket])
        {
            i--;
        }
        else
        {
            is_selected[pocket] = true;
        }
    }
    
    std::vector<size_t> selected_index;
    for(size_t i = 0; i < fitness.size(); i++){
        if(is_selected[i])
        {
            selected_index.push_back(i);
        }
    
    }
    return selected_index;
}

void Proposed(Dataset *dataset, size_t k)
{
    bool* is_minority = FindMinority(dataset);

    std::vector<size_t> majority_index;
    std::vector<size_t> minority_index;
    std::vector<float>  majority_fitness;
    std::vector<std::vector<size_t>> all_k_nearest_neighbors_index;
    size_t *num_rnn = (size_t*)malloc((dataset->training_set).size() * sizeof(size_t));
    memset(num_rnn, 0, (dataset->training_set).size() * sizeof(size_t));
    float *rnn_total_distance = (float*)malloc((dataset->training_set).size() * sizeof(size_t));
    memset(rnn_total_distance, 0, (dataset->training_set).size() * sizeof(size_t));

    for(size_t i = 0; i < (dataset->training_set).size(); i++)
    {
        size_t label = (dataset->training_set)[i][dataset->label_index];
        if(is_minority[label]) 
        {
            minority_index.push_back(i);
#ifdef USE_KNN
            continue;
#endif
        }
        else
        {
            majority_index.push_back(i);
        }

        // k_nearest_neighbors_index_with_distance[i] = {i'th nn index, distance from sample to its i'th nn}
        std::vector<std::vector<float>> k_nearest_neighbors_index_with_distance = FindKNearestNeighbors(dataset, (dataset->training_set)[i], k);
#ifdef USE_RNN
        if(is_minority[label]) 
        {
#ifdef CONCIDER_MAJ
            continue;
#endif
        }
        else
        {
#ifdef CONCIDER_MIN
            continue;
#endif
        }
        for(size_t j = 0; j < k; j++)
        {
            num_rnn[(size_t)k_nearest_neighbors_index_with_distance[j][0]]++;
            rnn_total_distance[(size_t)k_nearest_neighbors_index_with_distance[j][0]] += k_nearest_neighbors_index_with_distance[j][1];
        }
#endif
#ifdef USE_MNN
        std::vector<size_t> knn_index;
        for(size_t j = 0; j < k; j++)
        {
            knn_index.push_back((size_t)k_nearest_neighbors_index_with_distance[j][0]);
        }
        all_k_nearest_neighbors_index.push_back(knn_index);
        knn_index.clear();
#endif
#ifdef USE_KNN
        size_t maj_num = 0, min_num = 0;
        float total_distance = 0;
        for(size_t j = 0; j < k; j++)
        {
            size_t label = (dataset->training_set)[(size_t)k_nearest_neighbors_index_with_distance[j][0]][dataset->label_index];
            if(!is_minority[label])
            {
                maj_num++;
#if defined(CONSIDER_MAJ) || defined(KNN_DISTANCE)
                total_distance += k_nearest_neighbors_index_with_distance[j][1];
#endif
            }
            else
            {
                min_num++;
#if defined(CONSIDER_MIN) || defined(KNN_DISTANCE)
                total_distance += k_nearest_neighbors_index_with_distance[j][1];
#endif
            }
        }
#ifdef CONSIDER_MAJ
        float fitness = (maj_num == k) ? (k - 2) : (maj_num);
#endif
#ifdef CONSIDER_MIN
        float fitness = (min_num == 0) ? (1) : (min_num);
#endif

#ifdef DIVIDE_BY_DISTANCE
        fitness /= (float)(total_distance + 1);
#endif
#ifdef MULTIPLE_BY_DISTANCE
        fitness *= total_distance;
#endif
        majority_index.push_back(i);
        majority_fitness.push_back(fitness);
#endif
    }

#ifdef USE_MNN
    for(size_t i = 0; i < majority_index.size(); i++)
    {
        float total_distance = 0;
        size_t min_num_in_rnn = 0;
        for(size_t j = 0; j < k; j++)
        {
            size_t jth_nn_index = all_k_nearest_neighbors_index[majority_index[i]][j];
            size_t jth_nn_label = (dataset->training_set)[jth_nn_index][dataset->label_index];

#ifdef CONSIDER_MAJ
            if(is_minority[jth_nn_label])
            {
                continue;
            }
#endif
#ifdef CONSIDER_MIN
            if(!is_minority[jth_nn_label])
            {
                continue;
            }
#endif

            for(size_t l = 0; l < k; l++)
            {
                if(all_k_nearest_neighbors_index[jth_nn_index][l] == majority_index[i])
                {   
                    total_distance += EuclideanDistance((dataset->training_set)[jth_nn_index], (dataset->training_set)[majority_index[i]]);
                    min_num_in_rnn++;
                    break;
                }
            } 
        }
        // float fitness = (float)total_distance / (min_num_in_rnn + 1);
        float fitness = (min_num_in_rnn > k)? k : min_num_in_rnn;
#ifdef DIVIDE_BY_DISTANCE
        fitness /= ((float)total_distance + 1);
#endif
        // std::cout << fitness << ",";
        majority_fitness.push_back(min_num_in_rnn);
    }
#endif
#ifdef USE_RNN
    for(size_t i = 0; i < majority_index.size(); i++)
    {
        float fitness = num_rnn[majority_index[i]];
#ifdef DIVIDE_BY_DISTANCE
        fitness /= (rnn_total_distance[majority_index[i]] + 1);
#endif
        majority_fitness.push_back(fitness);
    }
#endif
    std::vector<size_t> selected_majority_index = RouletteWheelSelection(majority_fitness, minority_index.size());

    
    std::vector<std::vector<float>> preprocessed_data;
    for(size_t i = 0; i < selected_majority_index.size(); i++)
    {
        preprocessed_data.push_back((dataset->training_set)[majority_index[selected_majority_index[i]]]);
    }

    for(size_t i = 0; i < minority_index.size(); i++)
    {
        preprocessed_data.push_back((dataset->training_set)[minority_index[i]]);
    }
    
    (dataset->training_set).clear();
    dataset->training_set = preprocessed_data;
    GetDatasetInfo(dataset, true);
}

template<class T>
float CalculateDiversity(std::vector<std::vector<T>> &dataset, size_t n_classes)
{
    // Separate n-class dataset into n single-class datasets
    std::vector<std::vector<T>> class_dataset[n_classes + 1];
    for(unsigned int data_idx = 0, data_label_idx = dataset[0].size() - 1; data_idx < dataset.size(); data_idx++)
    {
        unsigned int data_label = dataset[data_idx][data_label_idx];
        class_dataset[data_label].push_back(dataset[data_idx]);
    }

    // Get representative data points by KMeansPP within each single-class dataset
    std::vector<std::vector<T>> representative_points;
    std::vector<unsigned int> representative_points_label;
    for(unsigned int class_idx = 1; class_idx <= n_classes; class_idx++)
    {   
        // Temporarily decided, waiting for experimentation
        size_t n_clusters = 0.1 * class_dataset[class_idx].size();
        n_clusters = (n_clusters == 0)? 1:n_clusters;
        
        // KMeansPP may produce centroids without any points; these centroids will be removed before returning
        std::vector<std::vector<T>> centroids = KMeansPP<T>(class_dataset[class_idx], n_clusters, 100, 1e-4);
        representative_points.insert(representative_points.end(), centroids.begin(), centroids.end());

        // centroids.size() <= n_clusters
        size_t n_valid_centroids = centroids.size();
        representative_points_label.insert(representative_points_label.end(), n_valid_centroids, class_idx);
    }

    // Turn representative points into a complete graph, where the weight is the distance between the source vertex and the destination vertex
    size_t n_representative_points = representative_points.size();
    std::vector<std::vector<float>> complete_graph_adjacency_matrix(n_representative_points, std::vector<float>(n_representative_points, 0));
    for(unsigned int src_vertex_idx = 0; src_vertex_idx < n_representative_points; src_vertex_idx++)
    {
        for(unsigned int dst_vertex_idx = 0; dst_vertex_idx < n_representative_points; dst_vertex_idx++)
        {
            complete_graph_adjacency_matrix[src_vertex_idx][dst_vertex_idx] = EuclideanDistance(representative_points[src_vertex_idx], representative_points[dst_vertex_idx]);
        }
    }

    // Get the MST of the complete graph formed by the representative points
    std::vector<std::vector<float>>mst_adjacency_matrix = Prim<float>(complete_graph_adjacency_matrix);
    complete_graph_adjacency_matrix.clear();

    // Calculate the ratio of neighboring vertices from different classes to the total number of neighboring vertices in the MST for each class
    std::vector<size_t> n_neighbors_per_class(n_classes + 1, 0);
    std::vector<size_t> n_diff_class_neighbors_per_class(n_classes + 1, 0);
#ifdef DEBUG
        std::cout << "Minimum Spanning Tree:" << std::endl;
#endif // DEBUG 
    for(unsigned int src_vertex_idx = 0; src_vertex_idx < n_representative_points; src_vertex_idx++)
    {
        unsigned int src_vertex_label = representative_points_label[src_vertex_idx];
        for(unsigned int dst_vertex_idx = 0; dst_vertex_idx < n_representative_points; dst_vertex_idx++)
        {
#ifdef DEBUG
            std::cout << mst_adjacency_matrix[src_vertex_idx][dst_vertex_idx] << " ";
#endif // DEBUG 
            if(mst_adjacency_matrix[src_vertex_idx][dst_vertex_idx] > 0)
            {
                unsigned int dst_vertex_label = representative_points_label[dst_vertex_idx];
                n_neighbors_per_class[src_vertex_label]++;
                if(src_vertex_label != dst_vertex_label)
                {
                    n_diff_class_neighbors_per_class[src_vertex_label]++;
                }
            }
        }
#ifdef DEBUG
        std::cout << std::endl;
#endif // DEBUG 
    }

    float diversity = 0;
#ifdef DEBUG
    std::cout << std::endl << "#different class neighbor, #neighbor" << std::endl;
#endif // DEBUG 
    for(unsigned int class_idx = 1; class_idx <= n_classes; class_idx++)
    {
#ifdef DEBUG
        std::cout << class_idx << ", [" << n_diff_class_neighbors_per_class[class_idx]  << ", " << n_neighbors_per_class[class_idx] << "]" << std::endl;
#endif
        diversity += (float)n_diff_class_neighbors_per_class[class_idx] / n_neighbors_per_class[class_idx];
    }
#ifdef DEBUG
    std::cout << "diversity: " << diversity << std::endl;
#endif
    return diversity;
}