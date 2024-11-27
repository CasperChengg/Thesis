#include "../inc/proposed.h"

static std::vector<uint32_t> CalculateClassCounts(const std::vector<std::vector<float>> &training_set, 
                                                    const uint32_t n_classes)
{
    const uint32_t training_label_idx = training_set[0].size() - 1;
    
    // Class labels start from 1, so n_classes requires (n_classes + 1) space for direct indexing
    std::vector<uint32_t> class_counts((n_classes + 1), 0); 

    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){   
        uint32_t training_data_label = training_set[training_data_idx][training_label_idx];
        class_counts[training_data_label]++;
    }

    return class_counts;
}


static float EuclideanDistance(const std::vector<float> &src, const std::vector<float> &dst)
{
    float square_distance = 0;
    // the last column of vector stores the label
    for(uint32_t attr_idx = 0; attr_idx < src.size() - 1; attr_idx++)
    {
        float diff = src[attr_idx] - dst[attr_idx];
        square_distance += diff * diff;
    }

    return sqrt(square_distance);
}

static bool isRelativeMinority(const Accuracies &training_set_accuracies, const std::vector<uint32_t>class_counts, 
                                const uint32_t compared_class_idx, const uint32_t comparator_class_idx)
{
    float false_rate_compared_to_comparator = (float)training_set_accuracies.confusion_matrix[comparator_class_idx][compared_class_idx]
                                                / class_counts[compared_class_idx];
    float false_rate_comparator_to_compared = (float)training_set_accuracies.confusion_matrix[compared_class_idx][comparator_class_idx]
                                                / class_counts[comparator_class_idx];
    
    if(false_rate_compared_to_comparator > false_rate_comparator_to_compared){
        return true;
    }
    else{
        return false;
    } 
}

static void CalculateSamplingWeights(const std::vector<std::vector<float>> &training_set, const Accuracies &training_set_accuracies, const std::vector<uint> &class_counts, const uint32_t k, std::vector<float> &sampling_weights)
{
    uint32_t training_data_label_idx = training_set[0].size() - 1;
    std::vector<uint32_t> minority_rnn_counts(training_set.size(), 0);
    std::vector<float> distances_to_minority_rnn(training_set.size(), 0.f);

    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){
        uint32_t training_data_label = training_set[training_data_idx][training_data_label_idx];

        auto compare = [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b){return a.second < b.second;};
        std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, decltype(compare)> k_nearest_neighbors(compare);
        for(uint32_t dst_data_idx = 0; dst_data_idx < training_set.size(); dst_data_idx++){
            k_nearest_neighbors.push({dst_data_idx, EuclideanDistance(training_set[training_data_idx], training_set[dst_data_idx])});
            if(k_nearest_neighbors.size() > k){
                k_nearest_neighbors.pop();
            }
        }

        // Total the number of Minority Reverse Nearest Neighbors (MRNN) and the distance to MRNN for each majority sample
        while(!k_nearest_neighbors.empty()){
            std::pair<uint32_t, float> nearest_neighbor = k_nearest_neighbors.top();
            uint32_t nearest_neighbor_idx   = nearest_neighbor.first;
            uint32_t nearest_neighbor_label = training_set[nearest_neighbor_idx][training_data_label_idx];
            if(isRelativeMinority(training_set_accuracies, class_counts, training_data_label, nearest_neighbor_label)){
                minority_rnn_counts[nearest_neighbor_idx]++;
                distances_to_minority_rnn[nearest_neighbor_idx] += nearest_neighbor.second;
            }
            k_nearest_neighbors.pop();
        }
    }
    
    sampling_weights.assign(training_set.size(), 0.f);
    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){
        uint32_t training_data_label = training_set[training_data_idx][training_data_label_idx];
        if(distances_to_minority_rnn[training_data_idx] == 0.f){
            sampling_weights[training_data_idx] = 0.f;
        }
        else{
            sampling_weights[training_data_idx] = minority_rnn_counts[training_data_idx] / distances_to_minority_rnn[training_data_idx];
        }
        // std::cout << training_data_idx << "," << sampling_weights[training_data_idx] << std::endl;
    }
}

static void RouletteWheelSelection(std::vector<bool> &selection_result, std::vector<float> &fitness, const uint32_t n_rounds)
{
    // Sum up the total fitness in order to do the sum-to-one normalization
    float total_fitness = std::accumulate(fitness.begin(), fitness.end(), 0.f);

    // Set up the random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Draw without replacement
    for(uint32_t round = 0; round < n_rounds; round++)
    {
        std::uniform_real_distribution<> distrib(0.0, total_fitness);
        float rand_0_total_fitnesss = distrib(gen);

        uint32_t selected_individual_idx;
        for(selected_individual_idx = 0; selected_individual_idx < fitness.size(); selected_individual_idx++){
            rand_0_total_fitnesss -= fitness[selected_individual_idx];
            if(rand_0_total_fitnesss <= 0){
                break;
            }
        }

        total_fitness -= fitness[selected_individual_idx];
        fitness[selected_individual_idx] = 0.f;
        selection_result[selected_individual_idx] = true;
    }
}

void Proposed(std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const uint32_t k, const ModelParameters model_parameters)
{
    std::vector<uint32_t> class_counts = CalculateClassCounts(training_set, n_classes);
    PDEBUG("[Dataset Overview]\n");
    PDEBUG("-Size             :%ld\n", training_set.size());
    PDEBUG("-Dimension        :%ld\n", training_set[0].size());
    PDEBUG("-Data Distribution:\n");
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        PDEBUG("\tClass %u: %u (%f %%)\n", class_idx, class_counts[class_idx], 
                                            (float)class_counts[class_idx] / training_set.size() * 100);
    }

    Accuracies training_set_accuracies = Validation(training_set, training_set, n_classes, model_parameters);

    std::vector<float> sampling_weights(training_set.size(), 0.f);
    CalculateSamplingWeights(training_set, training_set_accuracies, class_counts, k, sampling_weights);

    uint32_t n_removed_candidate = std::count_if(sampling_weights.begin(), sampling_weights.end(), 
                                                    [](float sampling_weight){return sampling_weight > 0.f;}); 
    uint32_t n_removed = (training_set.size() * training_set_accuracies.g_mean > n_removed_candidate)? 
                            n_removed_candidate : training_set.size() * training_set_accuracies.g_mean;

    PDEBUG("[Preprocessing Detail]\n");
    PDEBUG("-Num Removed Candidate : %u\n", n_removed_candidate);
    PDEBUG("-Sampling Size         : %u\n", n_removed);

    std::vector<bool> is_removed(training_set.size(), false);
    RouletteWheelSelection(is_removed, sampling_weights, n_removed);

    for(int training_data_idx = (training_set.size() - 1); training_data_idx >= 0; training_data_idx--){
        if(is_removed[training_data_idx]){
            training_set.erase(training_set.begin() + training_data_idx);
        }
    }

    FDEBUG(class_counts = CalculateClassCounts(training_set, n_classes));
    PDEBUG("[Preprocessing Summary]\n");
    PDEBUG("-Size             :%ld\n", training_set.size());
    PDEBUG("-Dimension        :%ld\n", training_set[0].size());
    PDEBUG("-Data Distribution:\n");
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        PDEBUG("\tClass %u: %u (%f %%)\n", class_idx, class_counts[class_idx], 
                                            (float)class_counts[class_idx] / training_set.size() * 100);
    }
}

