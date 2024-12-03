#include "../inc/k_means_pp.h"

static float EuclideanDistance(std::vector<float> &src, std::vector<float> &dst)
{
    const uint32_t n_dimension = src.size() - 1;
    float square_distance = 0;

    // the last column of vector stores the label, so we skip it
    for(uint32_t dim_idx = 0; dim_idx < n_dimension; dim_idx++)
    {
        float diff = src[dim_idx] - dst[dim_idx];
        square_distance += diff * diff;
    }

    return sqrt(square_distance);
}

static uint32_t RouletteWheelSelection(std::vector<float> fitnesses)
{
    float total_fitness = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, total_fitness);

    float random_value = distrib(gen);
    for(uint32_t fitness_idx = 0; fitness_idx < fitnesses.size(); fitness_idx++){
        random_value -= fitnesses[fitness_idx];
        if(random_value <= 0.f){
            return fitness_idx;
        }
    }

    return fitnesses.size() - 1;
}

static void SelectInitCentroids(std::vector<std::vector<float>> &training_set, uint32_t n_clusters, std::vector<std::vector<float>> &init_centroids)
{
    const uint32_t n_features = training_set[0].size() - 1;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, training_set.size() - 1);

    // KMeansPP select only existing data as initial centroids
    std::vector<uint32_t> init_centroids_idx(n_clusters); 

    init_centroids_idx[0] = distrib(gen); // Random select first centroid
    init_centroids.reserve(n_clusters);
    init_centroids.push_back(training_set[init_centroids_idx[0]]);

    std::vector<std::vector<float>> distance(training_set.size(), std::vector<float>(training_set.size()));
    for(uint32_t src_idx = 0; src_idx < training_set.size(); src_idx++){
        distance[src_idx][src_idx] = 0;
        for(uint32_t dst_idx = (src_idx + 1); dst_idx < training_set.size(); dst_idx++){
            float dist = EuclideanDistance(training_set[src_idx], training_set[dst_idx]);
            distance[src_idx][dst_idx] = dist;
            distance[dst_idx][src_idx] = dist;
        }
    }

    for(uint32_t centroid_idx = 1; centroid_idx < n_clusters; centroid_idx++){
        std::vector<float> fitnesses(training_set.size(), 0.f);
        for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){
            for(uint32_t _centroid_idx = 0; _centroid_idx < centroid_idx; _centroid_idx++){
                uint32_t dst_idx = init_centroids_idx[_centroid_idx];
                // set distance from src to selected centroids as fitness
                fitnesses[training_data_idx] += distance[training_data_idx][dst_idx];   
            }
        }

        uint32_t selected_idx = RouletteWheelSelection(fitnesses);

        init_centroids_idx[centroid_idx] = selected_idx;
        init_centroids.push_back(training_set[init_centroids_idx[centroid_idx]]);
    }
}

std::vector<std::vector<float>> KMeansPP(std::vector<std::vector<float>> &training_set, const uint32_t n_clusters,  const uint32_t max_iter, const float tolerance)
{
    const uint32_t n_features = training_set[0].size() - 1; // without label
    uint32_t n_iter = 0;

    std::vector<uint32_t> label(training_set.size(), 0);
    std::vector<uint32_t> cluster_counts(n_clusters, 0);

    float previous_SSE = 0, current_SSE = std::numeric_limits<float>::max();

    std::vector<std::vector<float>> centroids;
    SelectInitCentroids(training_set, n_clusters, centroids);

    do{
        previous_SSE = current_SSE;
        current_SSE  = 0;

        cluster_counts.assign(cluster_counts.size(), 0.f);
        for(uint32_t data_idx = 0; data_idx < training_set.size(); data_idx++){
            uint32_t nearest_centroid_idx = 0;
            float distance_to_nearest_centroid = std::numeric_limits<float>::max();
            for(uint32_t cluster_idx = 0; cluster_idx < n_clusters; cluster_idx++){
                float distance_to_centroid = EuclideanDistance(training_set[data_idx], centroids[cluster_idx]);
                if(distance_to_centroid < distance_to_nearest_centroid){
                    nearest_centroid_idx         = cluster_idx;
                    distance_to_nearest_centroid = distance_to_centroid;
                }
            }
            label[data_idx] = nearest_centroid_idx;
            current_SSE += distance_to_nearest_centroid * distance_to_nearest_centroid;
        }

        centroids.assign(n_clusters, std::vector<float>(n_features, 0.f));
        for(uint32_t data_idx = 0; data_idx < training_set.size(); data_idx++){  
            cluster_counts[label[data_idx]]++;
            for(uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++){
                centroids[label[data_idx]][feature_idx] += training_set[data_idx][feature_idx];
            }
        }

        for(uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++){  
            for(uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++){
                centroids[centroid_idx][feature_idx] /= cluster_counts[centroid_idx];
            }
        }

        if(++n_iter > max_iter){
            break;
        }
    }
    while(previous_SSE - current_SSE > tolerance);

    for(int centroid_idx = centroids.size() - 1; centroid_idx >= 0; centroid_idx--){
        if(std::isnan(centroids[centroid_idx][0])){
            centroids.erase(centroids.begin() + centroid_idx);
        }
    }

    return centroids;
}


