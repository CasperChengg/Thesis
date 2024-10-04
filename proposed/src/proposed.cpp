#include "../inc/proposed.h"
#define DEBUG

#pragma region FUNCTION_DECLARATION
template <typename T>
void CalculateClassCounts(std::vector<std::vector<T>>dataset, bool is_existing_data[], uint32_t class_counts[], uint32_t n_classes, uint32_t n_data_items);
template<typename T>
void SeperateMajMin(std::vector<std::vector<T>> &dataset, bool is_existing_data[], uint32_t class_counts[], uint32_t n_classes, bool is_majority[], bool is_minority[], uint32_t n_data_items);
template<typename T>
void CalculateFitnesses(std::vector<std::vector<T>> &dataset, bool is_existing_data[], bool is_majority[], fitness fitnesses[], uint32_t k, uint32_t n_data_items);
void RouletteWheelSelection(bool is_existing_data[], bool is_majority[], fitness fitnesses[], uint32_t n_retained_majority, uint32_t n_data_items);
template<typename T>
float CalculateDiversity(std::vector<std::vector<T>> &dataset, bool is_existing_data[], uint32_t n_data_items, uint32_t n_classes);
#pragma endregion // FUNCTION_DECLARATION

template<typename T>
void Proposed(std::vector<std::vector<T>>dataset, uint32_t n_classes, uint32_t k)
{
    // Original data size
    uint32_t n_data_items = dataset.size();

    // Store the number of samples per class
    uint32_t *class_counts = new uint32_t[n_classes + 1];

    // Store whether the data exists in the current iteration
    // is_existing_data = is_majority & is_minoritys
    bool *is_existing_data = new bool[n_data_items];
    memset(is_existing_data, true, n_data_items * sizeof(bool));
    
    // Store whether the data is majority in the current iteration
    bool *is_majority = new bool[n_data_items];
    memset(is_majority, false, n_data_items * sizeof(bool));

    // Store whether the data is minority in the current iteration
    bool *is_minority = new bool[n_data_items];
    memset(is_minority, false, n_data_items * sizeof(bool));
    
    // Store the fitness of data
    fitness *fitnesses = new fitness[n_data_items];

    for(uint32_t n_iterations = 0;;){
        
        // Calculate the number of samples per class
        CalculateClassCounts<T>(dataset, is_existing_data, class_counts, n_classes, n_data_items);

        // Decide minority classes based on the proposed method
        SeperateMajMin<T>(dataset, is_existing_data, class_counts, n_classes, is_majority, is_minority, n_data_items);
        
        uint32_t n_existing_data = 0;
        uint32_t n_existing_maj  = 0;
        uint32_t n_existing_min  = 0;
        for(uint32_t data_idx = 0; data_idx < n_data_items; data_idx++)
        {
            if(is_majority[data_idx])
            {
                n_existing_maj++;
            }
            else
            {
                n_existing_min++;
            }
        }
        n_existing_data = n_existing_maj + n_existing_min;

#ifdef DEBUG
        std::cout << "==========" << n_iterations << "'th iteration" << "==========" << std::endl;
        std::cout << "[ ";
        for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++)
        {
            std::cout << class_counts[class_idx] << " ";
        }
        std::cout << "]" << std::endl;
#endif // DEBUG

        // Calculate Majority Fitness
        CalculateFitnesses<T>(dataset, is_existing_data, is_majority, fitnesses, k, n_data_items);
        
        // Temporarily decided, waiting for experimentation
        uint32_t n_retained_majority = n_existing_min;
        if(n_retained_majority > n_existing_maj)
        {
            float diversity = CalculateDiversity(dataset, is_existing_data, n_data_items, n_classes);
#ifdef DEBUG
            std::cout << "diversity: " << diversity << std::endl;
#endif //DEBUG
            break;
        }
        else
        {
            RouletteWheelSelection(is_existing_data, is_majority, fitnesses, n_retained_majority, n_data_items);
        }
        
        
        float diversity = CalculateDiversity(dataset, is_existing_data, n_data_items, n_classes);

#ifdef DEBUG
        std::cout << "diversity: " << diversity << std::endl;
#endif // DEBUG

    }

    // Remove any non-existing data
    for(int data_idx = (n_data_items - 1); data_idx >= 0; data_idx--)
    {
        if(!is_existing_data[data_idx]){
            dataset.erase(dataset.begin() + data_idx);
        }
    }

    CalculateClassCounts<T>(dataset, is_existing_data, class_counts, n_classes, n_data_items);
#ifdef DEBUG
    std::cout << "-----Final Result-----" << std::endl;
    std::cout << "[ ";
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++)
    {
        std::cout << class_counts[class_idx] << " ";
    }
    std::cout << "]" << std::endl;
#endif // DEBUG

#pragma region FREEING_MEMORY
    delete []class_counts;
    delete []is_existing_data;
    delete []is_majority;
    delete []is_minority;
    delete []fitnesses;
#pragma endregion // FREEING_MEMORY

}

template<typename T>
void CalculateClassCounts(std::vector<std::vector<T>>dataset, bool is_existing_data[], uint32_t class_counts[], uint32_t n_classes, uint32_t n_data_items)
{
    if(class_counts == NULL)
    {
        printf("./%s:%d: error: null pointer\n", __FILE__, __LINE__);
        exit(1);
    }

    uint32_t data_label_idx = dataset[0].size() - 1;

    memset(class_counts, 0, (n_classes + 1) * sizeof(uint32_t));
    for(uint32_t data_idx = 0; data_idx < n_data_items; data_idx++)
    {   
        // Count only the existing data
        if(!is_existing_data[data_idx])
        {
            continue;
        }
        uint32_t data_label = dataset[data_idx][data_label_idx];
        class_counts[data_label]++;
    }
}

template<typename T>
void SeperateMajMin(std::vector<std::vector<T>> &dataset, bool is_existing_data[], uint32_t class_counts[], uint32_t n_classes, bool is_majority[], bool is_minority[], unsigned int n_data_items)
{
    if(class_counts == NULL)
    {
        printf("./%s:%d: error: null pointer\n", __FILE__, __LINE__);
        exit(1);
    }
    
    uint32_t data_label_idx = dataset[0].size() - 1;

    // Temporarily decided, waiting for experimentation
    // minority = [class_i, class_counts[i] <= avg_class_counts]
    // majority = [class_j, class_counts[j] >  avg_class_counts]
    float avg_class_counts = (float)dataset.size() / n_classes;

    memset(is_majority, false, n_data_items * sizeof(bool));
    memset(is_minority, false, n_data_items * sizeof(bool));
    
    for(uint32_t data_idx = 0; data_idx < n_data_items; data_idx++)
    {   
        // Count only the existing data
        if(!is_existing_data[data_idx])
        {
            continue;
        }

        uint32_t data_label = dataset[data_idx][data_label_idx];
        if(class_counts[data_label] <= avg_class_counts)
        {
            is_minority[data_idx] = true; 
        }
        else
        {
            is_majority[data_idx] = true;
        }
    }
}

float EuclideanDistance(std::vector<float> &src, std::vector<float> &dst)
{
    float square_distance = 0;
    // the last column of vector stores the label
    for(uint32_t i = 0; i < src.size() - 1; i++)
    {
        square_distance += pow(src[i] - dst[i], 2);
    }
    return sqrt(square_distance);
}

template<typename T>
void CalculateFitnesses(std::vector<std::vector<T>> &dataset, bool is_existing_data[], bool is_majority[], fitness fitnesses[], uint32_t k, uint32_t n_data_items)
{
    uint32_t data_label_idx = dataset[0].size() - 1;
    memset(fitnesses, 0, n_data_items * sizeof(fitness));

    for(uint32_t data_idx = 0; data_idx < n_data_items; data_idx++)
    {
        if(!(is_existing_data[data_idx]) || !(is_majority[data_idx]))
        {
            continue;
        }
    
        // Find K Nearest Neighbors
        std::vector<std::pair<uint32_t, float>> distance;
        for(uint32_t dst_data_idx = 0; dst_data_idx < n_data_items; dst_data_idx++)
        {
            if((dst_data_idx == data_idx) || (!is_existing_data[dst_data_idx]))
            {
                continue;
            }
            distance.push_back(std::make_pair(dst_data_idx, EuclideanDistance(dataset[data_idx], dataset[dst_data_idx])));
        }
        sort(distance.begin(), distance.end(), [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b ){return a.second < b.second;});
        
        // Total the number of Minority Reverse Nearest Neighbors (MRNN) and the distance to MRNN for each majority sample
        for(uint32_t distance_idx = 0; distance_idx < k; distance_idx++)
        {
            uint32_t nn_idx = distance[distance_idx].first;
            if(is_majority[nn_idx])
            {
                continue;
            }
            fitnesses[nn_idx].minority_rnn_counts++;
            fitnesses[nn_idx].distance_to_minority_rnn += distance[distance_idx].second;
        }
    }

    for(uint32_t data_idx = 0; data_idx < n_data_items; data_idx++)
    {
        if((!is_existing_data[data_idx]) || (!is_majority[data_idx]))
        {
            continue;
        }
        fitnesses[data_idx].fitness = fitnesses[data_idx].minority_rnn_counts * fitnesses[data_idx].distance_to_minority_rnn;
    }

}

void RouletteWheelSelection(bool is_existing_data[], bool is_majority[], fitness fitnesses[], uint32_t n_retained_majority, uint32_t n_data_items)
{
    // Sum up the total fitness in order to do the sum-to-one normalization
    float total_fitness = 0.0;
    for(uint32_t data_idx = 0; data_idx < n_data_items; data_idx++)
    {
        if(is_majority[data_idx])
        {
            total_fitness += fitnesses[data_idx].fitness;
        }
    } 

    // Calculate the upper limit of each pocket in the roulette wheel
    float *pocket_limits = new float[n_data_items];
    memset(pocket_limits, 0, n_data_items * sizeof(float));

    uint32_t last_nonzero_pocket_idx = 0;
    // Sum-to-one Normalization
    pocket_limits[0] = fitnesses[0].fitness / total_fitness; 
    for(uint32_t pocket_idx = 1; pocket_idx < n_data_items; pocket_idx++)
    {
        if(pocket_limits[pocket_idx] == 0)
        {
            continue;
        }
        pocket_limits[pocket_idx] = pocket_limits[pocket_idx - 1] + fitnesses[pocket_idx].fitness / total_fitness;
        last_nonzero_pocket_idx = pocket_idx;
    }
    pocket_limits[last_nonzero_pocket_idx] = 1.0;

    // Set up the random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    // Remove the existance of all the majority samples
    for(uint32_t data_idx = 0; data_idx < n_data_items; data_idx++)
    {
        is_existing_data[data_idx] = !is_majority[data_idx] & is_existing_data[data_idx];
    }
    
    bool *is_non_selected_majority = new bool[n_data_items];
    memcpy(is_non_selected_majority, is_majority, n_data_items * sizeof(bool));
    
    // Draw without replacement
    for(uint32_t selected_item_idx = 0; selected_item_idx < n_retained_majority; selected_item_idx++)
    {
        float rand_0_1 = distrib(gen);

        uint32_t selected_pocket_idx;
        for(selected_pocket_idx = 0; selected_pocket_idx < n_data_items; selected_pocket_idx++)
        {
            if(rand_0_1 <= pocket_limits[selected_item_idx])
            {
                break;
            }
        }

        if(is_non_selected_majority[selected_pocket_idx])
        {
            is_non_selected_majority[selected_pocket_idx] = false;
            is_existing_data[selected_pocket_idx] = true;
        }
        else
        {
            selected_item_idx--;
        }
    }

    delete []pocket_limits;
    delete []is_non_selected_majority;
}

template<class T>
float CalculateDiversity(std::vector<std::vector<T>> &dataset, bool is_existing_data[], uint32_t n_data_items, uint32_t n_classes)
{
    // Separate n-class dataset into n single-class datasets
    std::vector<std::vector<T>> class_dataset[n_classes + 1];
    for(uint32_t data_idx = 0, data_label_idx = dataset[0].size() - 1; data_idx < n_data_items; data_idx++)
    {
        if(!is_existing_data[data_idx])
        {
            continue;
        }
        uint32_t data_label = dataset[data_idx][data_label_idx];
        class_dataset[data_label].push_back(dataset[data_idx]);
    }

    // Get representative data points by KMeansPP within each single-class dataset
    std::vector<std::vector<T>> representative_points;
    std::vector<uint32_t> representative_points_label;
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++)
    {   
        // Temporarily decided, waiting for experimentation
        uint32_t n_clusters = 0.1 * class_dataset[class_idx].size();
        n_clusters = (n_clusters == 0)? 1:n_clusters;
        
        // KMeansPP may produce centroids without any points; these centroids will be removed before returning
        std::vector<std::vector<T>> centroids = KMeansPP<T>(class_dataset[class_idx], n_clusters, 100, 1e-4);
        representative_points.insert(representative_points.end(), centroids.begin(), centroids.end());

        // centroids.size() <= n_clusters
        uint32_t n_valid_centroids = centroids.size();
        representative_points_label.insert(representative_points_label.end(), n_valid_centroids, class_idx);
    }

    // Turn representative points into a complete graph, where the weight is the distance between the source vertex and the destination vertex
    uint32_t n_representative_points = representative_points.size();
    std::vector<std::vector<float>> complete_graph_adjacency_matrix(n_representative_points, std::vector<float>(n_representative_points, 0));
    for(uint32_t src_vertex_idx = 0; src_vertex_idx < n_representative_points; src_vertex_idx++)
    {
        for(uint32_t dst_vertex_idx = 0; dst_vertex_idx < n_representative_points; dst_vertex_idx++)
        {
            complete_graph_adjacency_matrix[src_vertex_idx][dst_vertex_idx] = EuclideanDistance(representative_points[src_vertex_idx], representative_points[dst_vertex_idx]);
        }
    }

    // Get the MST of the complete graph formed by the representative points
    std::vector<std::vector<float>>mst_adjacency_matrix = Prim<float>(complete_graph_adjacency_matrix);
    complete_graph_adjacency_matrix.clear();

    // Calculate the ratio of neighboring vertices from different classes to the total number of neighboring vertices in the MST for each class
    std::vector<uint32_t> n_neighbors_per_class(n_classes + 1, 0);
    std::vector<uint32_t> n_diff_class_neighbors_per_class(n_classes + 1, 0);

    for(uint32_t src_vertex_idx = 0; src_vertex_idx < n_representative_points; src_vertex_idx++)
    {
        uint32_t src_vertex_label = representative_points_label[src_vertex_idx];
        for(uint32_t dst_vertex_idx = 0; dst_vertex_idx < n_representative_points; dst_vertex_idx++)
        {
            if(mst_adjacency_matrix[src_vertex_idx][dst_vertex_idx] > 0)
            {
                uint32_t dst_vertex_label = representative_points_label[dst_vertex_idx];
                n_neighbors_per_class[src_vertex_label]++;
                if(src_vertex_label != dst_vertex_label)
                {
                    n_diff_class_neighbors_per_class[src_vertex_label]++;
                }
            }
        }
    }

    float diversity = 0;
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++)
    {
        diversity += (float)n_diff_class_neighbors_per_class[class_idx] / n_neighbors_per_class[class_idx];
    }
    return diversity;
}

template void Proposed(std::vector<std::vector<float>>dataset, uint32_t n_classes, uint32_t k);