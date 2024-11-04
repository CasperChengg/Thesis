#include "../inc/k_means_pp.h"

#pragma region FUNCTION_DECLARATION
static float EuclideanDistance(std::vector<float> &src, std::vector<float> &dst);
static uint32_t RouletteWheelSelection(std::vector<float> fitnesses);
std::vector<std::vector<float>> SelectInitCentroids(std::vector<std::vector<float>> &dataset, uint32_t n_clusters);
#pragma endregion // FUNCTION_DECLARATION

std::vector<std::vector<float>> KMeansPP(std::vector<std::vector<float>> &dataset, uint32_t n_clusters,  uint32_t max_iter, float tolerance)
{
    uint32_t dimension = dataset[0].size() - 1, n_iter = 0;

    // label of each sample
    std::vector<uint32_t> label(dataset.size(), 0);

    // number of samples within each cluster
    std::vector<uint32_t> n_samples_in_cluster(n_clusters, 0);

    // for calculating SSE differnece between iterations
    float previous_SSE = 0, current_SSE = std::numeric_limits<float>::max();

    // find initial centroids 
    std::vector<std::vector<float>> centroids = SelectInitCentroids(dataset, n_clusters);

#ifdef DEBUG
    std::cout << n_clusters << ", " << tolerance << std::endl;
    for(uint32_t i = 0; i < n_clusters; i++)
    {
        for(uint32_t j = 0; j < dimension; j++)
        {
            std::cout << centroids[i][j] << " ";
        }
        std::cout << std::endl;
    }
#endif
    
    do
    {
        // find the nearest centroid of each sample
        for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++)
        {
            uint32_t nearest_centroid_idx = 0;
            float distance_to_nearest_centroid = std::numeric_limits<float>::max();
            for(uint32_t cluster_idx = 0; cluster_idx < n_clusters; cluster_idx++)
            {
                float distance_to_centroid = EuclideanDistance(dataset[data_idx], centroids[cluster_idx]);
                if(distance_to_centroid < distance_to_nearest_centroid)
                {
                    nearest_centroid_idx         = cluster_idx;
                    distance_to_nearest_centroid = distance_to_centroid;
                }
            }
            label[data_idx] = nearest_centroid_idx;
        }

        // calcualte SSE
        previous_SSE = current_SSE;
        current_SSE  = 0;
        for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++)
        {   
            // square distance to the belonging centroid
            float error = EuclideanDistance(dataset[data_idx], centroids[label[data_idx]]);
            current_SSE += error * error;
        }

        for(uint32_t centroid_idx = 0;  centroid_idx < n_clusters;  centroid_idx++)
        {
            for(uint32_t dim_idx = 0; dim_idx < dimension; dim_idx++)
            {
                centroids[centroid_idx][dim_idx] = 0;
            }
        }

        n_samples_in_cluster.assign(n_samples_in_cluster.size(), 0.f);
        for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++)
        {  
            n_samples_in_cluster[label[data_idx]]++;
            for(uint32_t dim_idx = 0; dim_idx < dimension; dim_idx++)
            {
                centroids[label[data_idx]][dim_idx] += dataset[data_idx][dim_idx];
            }
        }

        for(uint32_t centroid_idx = 0; centroid_idx < n_clusters; centroid_idx++)
        {  
            for(uint32_t dim_idx = 0; dim_idx < dimension; dim_idx++)
            {
                centroids[centroid_idx][dim_idx] /= n_samples_in_cluster[centroid_idx];
            }
        }
#ifdef DEBUG
        for(uint32_t i = 0; i < n_clusters; i++)
        {
            for(uint32_t j = 0; j < dimension; j++)
            {
                std::cout << centroids[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << n_iter << "'th iteration, SSE = " << current_SSE << std::endl;
        std::cout << "====================" << std::endl;
#endif

        if(++n_iter > max_iter)
        {
            break;
        }
    }
    while(previous_SSE - current_SSE > tolerance);

    for(int centroid_idx = (centroids.size() - 1); centroid_idx >= 0; centroid_idx--)
    {
        if(std::isnan(centroids[centroid_idx][0]))
        {
            centroids.erase(centroids.begin() + centroid_idx);
        }
    }

    return centroids;
}

static float EuclideanDistance(std::vector<float> &src, std::vector<float> &dst)
{
    float square_distance = 0;

    // the last column of vector stores the label, so we skip it
    for(uint32_t i = 0; i < src.size() - 1; i++)
    {
        float diff = src[i] - dst[i];
        square_distance += diff * diff;
    }

    return sqrt(square_distance);
}
std::vector<std::vector<float>> SelectInitCentroids(std::vector<std::vector<float>> &dataset, uint32_t n_clusters)
{
    std::vector<uint32_t> init_centroids_idx(n_clusters);
    std::vector<std::vector<float>> init_centroids(n_clusters, std::vector<float>(dataset[0].size()));
    std::vector<std::vector<float>> distance(dataset.size(), std::vector<float>(dataset.size()));

    for(uint32_t src_idx = 0; src_idx < dataset.size(); src_idx++)
    {
        distance[src_idx][src_idx] = 0;
        for(uint32_t dst_idx = (src_idx + 1); dst_idx < dataset.size(); dst_idx++)
        {
            float dist = EuclideanDistance(dataset[src_idx], dataset[dst_idx]);
            distance[src_idx][dst_idx] = dist;
            distance[dst_idx][src_idx] = dist;
        }
    }

    // Set the densest point as the first initial centroid
    uint32_t min_total_distance_point_idx = 0;
    float min_total_distance = std::numeric_limits<float>::max();
    
    for (uint32_t dst_idx = 0; dst_idx < dataset.size(); dst_idx++) {
        float total_distance = 0.f;
        for (uint32_t src_idx = 0; src_idx < dataset.size(); src_idx++) {
            total_distance += distance[src_idx][dst_idx];
        }
        if (total_distance < min_total_distance) {
            min_total_distance = total_distance;
            min_total_distance_point_idx = dst_idx;
        }
    }
    init_centroids_idx[0] = min_total_distance_point_idx;
    init_centroids[0] = dataset[min_total_distance_point_idx];
    
    for(uint32_t n_centroids = 1; n_centroids < n_clusters; n_centroids++)
    {
        std::vector<float> fitnesses(dataset.size(), 0.f);

        // Calculate fitness by the total distance to the existing centroids
        for(uint32_t src_idx = 0; src_idx < dataset.size(); src_idx++)
        {
            for(uint32_t centroid_idx = 0; centroid_idx < n_centroids; centroid_idx++)
            {
                uint32_t dst_idx = init_centroids_idx[centroid_idx];
                fitnesses[src_idx] += distance[src_idx][dst_idx];   
            }
        }

        // Select by fitness using Roulette Wheel Selection
        uint32_t selected_index = RouletteWheelSelection(fitnesses);

        init_centroids_idx[n_centroids] = selected_index;
        init_centroids[n_centroids] = dataset[selected_index];
    }

    return init_centroids;
}
static uint32_t RouletteWheelSelection(std::vector<float> fitnesses)
{
    // Calculate total fitness
    float total_fitness = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.f);
    if(total_fitness <= 0.f)
    {
        printf("./%s:%d: error: error fitness value\n", __FILE__, __LINE__);
        exit(1);
    }

    // Setting random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    // Perform Roulette wheel selection
    float rand_0_1 = distrib(gen);
    for(uint32_t fitness_idx = 0; fitness_idx < fitnesses.size(); fitness_idx++)
    {
        rand_0_1 -= fitnesses[fitness_idx] / total_fitness;
        if(rand_0_1 <= 0)
        {
            return fitness_idx;
        }
    }
    
    // In case of floating-point precision errors, return the last index
    return fitnesses.size() - 1;
}

