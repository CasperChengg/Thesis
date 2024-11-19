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

static std::vector<float> CalculateMinorityScores(const Accuracies &training_set_accuracies, const uint32_t n_classes)
{
    std::vector<float> majority_scores(n_classes + 1, 0.f);

    // float max_diversity = 0.f, min_diversity = std::numeric_limits<float>::max();
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
        // std::cout << training_set_accuracies.FPR[class_idx] << " " << training_set_accuracies.FNR[class_idx] << std::endl;                                   
        majority_scores[class_idx] = (training_set_accuracies.FPR[class_idx]) / (training_set_accuracies.FNR[class_idx] + 1);
        // if(diversity[class_idx] > max_diversity){
        //     max_diversity = diversity[class_idx];
        // }
        // if(diversity[class_idx] < min_diversity){
        //     min_diversity = diversity[class_idx];
        // }
    }

    // if(max_diversity == min_diversity){
    //     for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
    //         diversity[class_idx] = 1;
    //     }
    // }
    // else{
    //     for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
    //         diversity[class_idx] = (diversity[class_idx] - min_diversity) / (max_diversity - min_diversity);
    //     }
    // }

    return majority_scores;
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

static void CalculateSamplingWeights(const std::vector<std::vector<float>> &training_set, const std::vector<float> &majority_scores, const uint32_t k, std::vector<float> &sampling_weights)
{
    uint32_t training_data_label_idx = training_set[0].size() - 1;
    std::vector<uint32_t> minority_rnn_counts(training_set.size(), 0);
    std::vector<float> distances_to_minority_rnn(training_set.size(), 0.f);

    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){
        uint32_t training_data_label = training_set[training_data_idx][training_data_label_idx];
        if(majority_scores[training_data_label] == 0.f){
            continue;
        }

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
            if((majority_scores[training_data_label] < majority_scores[nearest_neighbor_label])){
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
            sampling_weights[training_data_idx] = minority_rnn_counts[training_data_idx] / distances_to_minority_rnn[training_data_idx] * majority_scores[training_data_label];
        }
        // std::cout << training_data_idx << "," << sampling_weights[training_data_idx] << std::endl;
    }
}

static void RouletteWheelSelection(std::vector<bool> &selection_result, std::vector<float> fitness, const uint32_t n_rounds)
{
    // Sum up the total fitness in order to do the sum-to-one normalization
    float total_fitness = std::accumulate(fitness.begin(), fitness.end(), 0.f);

    // Calculate the upper limit of each pocket in the roulette wheel
    std::vector<float> cumulative_prob(fitness.size(), 0.f);

    // Sum-to-one Normalization
    uint32_t last_nonzero_prob_idx = 0;
    cumulative_prob[0] = fitness[0] / total_fitness; 
    for(uint32_t fitness_idx = 1; fitness_idx < fitness.size(); fitness_idx++){
        cumulative_prob[fitness_idx] = cumulative_prob[fitness_idx - 1] + fitness[fitness_idx] / total_fitness;
        // std::cout << fitness[fitness_idx] << std::endl;
        if(fitness[fitness_idx] == 0.f){
            last_nonzero_prob_idx = fitness_idx;
        }
    }
    cumulative_prob[last_nonzero_prob_idx] = 1.0;

    // Set up the random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);
    
    // Draw without replacement
    for(uint32_t round = 0; round < n_rounds; round++)
    {
        float rand_0_1 = distrib(gen);
        uint32_t selected_individual_idx;
        
        for(selected_individual_idx = 0; selected_individual_idx < fitness.size(); selected_individual_idx++){
            // std::cout << cumulative_prob[selected_individual_idx] << std::endl;
            if(rand_0_1 <= cumulative_prob[selected_individual_idx]){
                break;
            }
        }
        // std::cout << selected_individual_idx << std::endl;
        if(!selection_result[selected_individual_idx]){
            selection_result[selected_individual_idx] = true;
        }
        else{
            round--;
        }
    }
}

void Proposed(std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const uint32_t k, const ModelParameters model_parameters)
{
    
    Accuracies training_set_accuracies = Validation(training_set, training_set, n_classes, model_parameters);

    std::vector<float> majority_scores  = CalculateMinorityScores(training_set_accuracies, n_classes);
    std::vector<float> sampling_weights(training_set.size(), 0.f);
    CalculateSamplingWeights(training_set, majority_scores, k, sampling_weights);
    
    uint32_t n_removed_candidate = 0;
    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){
        if(sampling_weights[training_data_idx] > 0.f){
            n_removed_candidate++;
        }
    }

    #ifdef DEBUG
        std::vector<uint32_t> class_counts = CalculateClassCounts(training_set, n_classes);
        std::cout << "[Dataset Overview]"  << std::endl;
        std::cout << "-Size             :" << training_set.size()    << std::endl;
        std::cout << "-Dimension        :" << training_set[0].size() << std::endl;
        std::cout << "-Data Distribution:" << std::endl;
        for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
            std::cout << "\tClass" << class_idx << ": " << class_counts[class_idx];
            std::cout << "(" << (float)class_counts[class_idx] / training_set.size() * 100 << "%, ";
            std::cout << majority_scores[class_idx] << ")" << std::endl; 
        }
        std::cout << std::endl;
    #endif // DEBUG

    std::vector<bool> is_removed(training_set.size(), false);
    uint32_t n_removed = (float)training_set.size() * training_set_accuracies.macro_FPR;
    if(n_removed > n_removed_candidate){
        n_removed = n_removed_candidate;
    }

    #ifdef DEBUG
        std::cout << "[Preprocessing Detail]" << std::endl;
        std::cout << "-Macro FPR             :" <<  training_set_accuracies.macro_FPR << std::endl;
        std::cout << "-Num Removed Candidate :" <<  n_removed_candidate  << std::endl;
        std::cout << "-Sampling Size         :" <<  n_removed  << std::endl << std::endl;
    #endif

    RouletteWheelSelection(is_removed, sampling_weights, n_removed);

    for(int training_data_idx = (training_set.size() - 1); training_data_idx >= 0; training_data_idx--){
        if(is_removed[training_data_idx]){
            training_set.erase(training_set.begin() + training_data_idx);
        }
    }

    #ifdef DEBUG
        class_counts = CalculateClassCounts(training_set, n_classes);
        training_set_accuracies = Validation(training_set, training_set, n_classes, model_parameters);
        majority_scores  = CalculateMinorityScores(training_set_accuracies, n_classes);
        std::cout << "[Preprocessing Summary]" << std::endl;
        std::cout << "-Size             :" << training_set.size()    << std::endl;
        std::cout << "-Dimension        :" << training_set[0].size() << std::endl;
        std::cout << "-Data Distribution:" << std::endl;
        for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
            std::cout << "\tClass" << class_idx << ": " << class_counts[class_idx];
            std::cout << "(" << (float)class_counts[class_idx] / training_set.size() * 100 << "%, ";
            std::cout << majority_scores[class_idx] << ")" << std::endl; 
        }
    #endif // DEBUG
}

