#include "../inc/k_means_pp.h"
static float EuclideanDistance(std::vector<float> &src, std::vector<float> &dst)
{
    float square_distance = 0;

    // the last column of vector stores the label, so we skip it
    for(size_t i = 0; i < src.size() - 1; i++)
    {
        float diff = src[i] - dst[i];
        square_distance += diff * diff;
    }

    return sqrt(square_distance);
}

static unsigned int RouletteWheelSelection(std::vector<float> fitnesses)
{
    size_t n_nonzero = 0;
    float total_fitness = 0.f;

    for(size_t i = 0; i < fitnesses.size(); i++)
    {
        if(fitnesses[i] > 0)
        {
            n_nonzero++;
            total_fitness += fitnesses[i];
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    if(n_nonzero == 0)
    {
        std::uniform_int_distribution<> index_distrib(0, fitnesses.size() - 1);
        return index_distrib(gen);
    }

    for(size_t pocket = 0; pocket < fitnesses.size(); pocket++)
    {
        fitnesses[pocket] /= total_fitness;
    }

    float rand_0_1 = distrib(gen);
    for(unsigned int i = 0; i < fitnesses.size(); i++)
    {
        rand_0_1 -= fitnesses[i];
        if(rand_0_1 <= 0)
        {
            return i;
        }
    }
    return fitnesses.size() - 1;
}

template <class T>
std::vector<std::vector<T>> SelectInitCentroids(std::vector<std::vector<T>> &dataset, size_t n_clusters)
{
    std::vector<std::vector<T>> init_centroids;
    unsigned int n_samples = dataset.size();

    // random first centroid
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, n_samples - 1);
    init_centroids.push_back(dataset[distrib(gen)]);
    
    for(unsigned int n_centroids = 0; n_centroids < (n_clusters - 1); n_centroids++)
    {
        std::vector<float> fitnesses(n_samples, 0);
        for(unsigned int i = 0; i < n_samples; i++)
        {
            fitnesses[i] += EuclideanDistance(dataset[i], init_centroids[n_centroids]);
        }

        // select by fitness
        unsigned int selected_index = RouletteWheelSelection(fitnesses);
        init_centroids.push_back(dataset[selected_index]);
    }

    return init_centroids;
}

template <class T>
std::vector<std::vector<T>> KMeansPP(std::vector<std::vector<T>> &dataset, size_t n_clusters,  size_t max_iter, float tolerance)
{
    size_t n_samples = dataset.size(), dimension = dataset[0].size() - 1, n_iter = 0;

    // label of each sample
    unsigned int *label = new unsigned int[n_samples];
    memset(label, 0, n_samples * sizeof(unsigned int)); 

    // number of samples within each cluster
    size_t *n_samples_in_cluster = new size_t[n_clusters];
    memset(n_samples_in_cluster, 0, n_clusters * sizeof(size_t));

    // for calculating SSE differnece between iterations
    float previous_SSE = 0, current_SSE = std::numeric_limits<float>::max();

    // find initial centroids 
    std::vector<std::vector<T>> centroids = SelectInitCentroids<T>(dataset, n_clusters);

#ifdef DEBUG
    std::cout << n_clusters << ", " << tolerance << std::endl;
    for(unsigned int i = 0; i < n_clusters; i++)
    {
        for(unsigned int j = 0; j < dimension; j++)
        {
            std::cout << centroids[i][j] << " ";
        }
        std::cout << std::endl;
    }
#endif
    
    do
    {
        // find the nearest centroid of each sample
        for(unsigned int i = 0; i < n_samples; i++)
        {
            unsigned int nearest_centroid_id   = 0;
            float distance_to_nearest_centroid = std::numeric_limits<float>::max();
            for(unsigned int j = 0; j < n_clusters; j++)
            {
                float distance_to_centroid = EuclideanDistance(dataset[i], centroids[j]);
                if(distance_to_centroid < distance_to_nearest_centroid)
                {
                    nearest_centroid_id          = j;
                    distance_to_nearest_centroid = distance_to_centroid;
                }
            }
            label[i] = nearest_centroid_id;
        }

        // calcualte SSE
        previous_SSE = current_SSE;
        current_SSE  = 0;
        for(unsigned int i = 0; i < n_samples; i++)
        {   
            // square distance to the belonging centroid
            float error = EuclideanDistance(dataset[i], centroids[label[i]]);
            current_SSE += error * error;
        }

        memset(n_samples_in_cluster, 0, n_clusters * sizeof(size_t));
        for(unsigned int i = 0; i < n_clusters; i++)
        {
            for(unsigned int j = 0; j < dimension; j++)
            {
                centroids[i][j] = 0;
            }
        }
        for(unsigned int i = 0; i < n_samples; i++)
        {  
            n_samples_in_cluster[label[i]]++;
            for(unsigned int j = 0; j < dimension; j++)
            {
                centroids[label[i]][j] += dataset[i][j];
            }
        }
        for(unsigned int i = 0; i < n_clusters; i++)
        {  
            for(unsigned int j = 0; j < dimension; j++)
            {
                centroids[i][j] /= n_samples_in_cluster[i];
            }
        }
#ifdef DEBUG
        for(unsigned int i = 0; i < n_clusters; i++)
        {
            for(unsigned int j = 0; j < dimension; j++)
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
            delete [] n_samples_in_cluster;
            break;
        }
    }
    while(previous_SSE - current_SSE > tolerance);

    for(unsigned int i = 0; i < centroids.size(); i++)
    {
        if(std::isnan(centroids[i][0]))
        {
            centroids.erase(centroids.begin() + i);
            i--;
        }
    }

    delete [] label;
    delete [] n_samples_in_cluster;

    return centroids;
}

template std::vector<std::vector<float>> KMeansPP(std::vector<std::vector<float>> &dataset, size_t n_clusters,  size_t max_iter, float tolerance);


