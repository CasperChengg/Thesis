#include <cmath>
#include <ctime>
#include <limits>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>

float SquareEuclideanDistance(std::vector<float> &src, std::vector<float> &dst)
{
    float square_distance = 0;

    // the last column of vector stores the label
    for(size_t i = 0; i < src.size() - 1; i++)
    {
        square_distance += pow(src[i] - dst[i], 2);
    }

    return sqrt(square_distance);
}

unsigned int BinarySearchRange(std::vector<float> array, float target)
{
    unsigned int start = 0, end = array.size() - 1;
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
        unsigned int middle = start + (end - start) / 2;
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

unsigned int RouletteWheelSelection(std::vector<float> fitnesses)
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

    srand(time(NULL));
    if(n_nonzero == 0)
    {
        return rand() % (fitnesses.size());
    }

    fitnesses[0] /= total_fitness;
    for(size_t pocket = 1; pocket < fitnesses.size(); pocket++)
    {
        fitnesses[pocket] = fitnesses[pocket - 1] + fitnesses[pocket] / total_fitness;
    }


    float hit_point             = (float)rand() / RAND_MAX;
    unsigned int selected_index = BinarySearchRange(fitnesses, hit_point);
    
    return selected_index;
}

template <class T>
std::vector<std::vector<T>> SelectInitCentroids(std::vector<std::vector<T>> &dataset, size_t n_clusters)
{
    std::vector<std::vector<T>> init_centroids;
    unsigned int n_samples = dataset.size();

    // random first centroid
    srand(time(NULL));
    init_centroids.push_back(dataset[rand() % n_samples]);
    
    for(unsigned int n_centroids = 1; n_centroids < n_clusters; n_centroids++)
    {
        std::vector<float> fitnesses;
        for(unsigned int i = 0; i < n_samples; i++)
        {
            float fitness = 0;
            for(unsigned int j = 0; j < n_centroids; j++)
            {
                // total distance from sample to centroids
                fitness += SquareEuclideanDistance(dataset[i], init_centroids[j]);
            }
            fitnesses.push_back(fitness);
        }

        // select by fitness
        unsigned int selected_index = RouletteWheelSelection(fitnesses);
        init_centroids.push_back(dataset[selected_index]);
    }

    return init_centroids;
}

template <class T>
unsigned int *KMeansPP(std::vector<std::vector<T>> &dataset, size_t n_clusters,  size_t max_iter, float tolerance)
{
    size_t n_samples = dataset.size(), n_iter = 0;

    // label of each sample
    unsigned int *label = new unsigned int[n_samples];
    memset(label, 0, n_samples * sizeof(unsigned int)); 

    // for calculating SSE differnece between iterations
    float previous_SSE = 0, current_SSE = std::numeric_limits<float>::max(); 
    
    float *SSE = new float[n_clusters];
    std::vector<std::vector<T>> centroids = SelectInitCentroids<T>(dataset, n_clusters);
    
    while(current_SSE - previous_SSE > tolerance)
    {
        // find the nearest centroid of each sample
        for(unsigned int i = 0; i < n_samples; i++)
        {
            unsigned int nearest_centroid_id   = 0;
            float nearest_distance_to_centroid = std::numeric_limits<float>::max();
            for(unsigned int j = 0; j < n_clusters; j++)
            {
                float distance_to_centroid = SquareEuclideanDistance(dataset[i], centroids[j]);
                if(distance_to_centroid < nearest_distance_to_centroid)
                {
                    nearest_centroid_id          = j;
                    nearest_distance_to_centroid = distance_to_centroid;
                }
            }
            label[i] = nearest_centroid_id;
        }

        // calcualte SSE
        memset(SSE, 0, n_clusters * sizeof(float));
        for(unsigned int i = 0; i < n_samples; i++)
        {   
            // distance to the belonging centroid
            float square_error = SquareEuclideanDistance(dataset[i], centroids[label[i]]);
            SSE[label[i]] += square_error;
        }

        previous_SSE = current_SSE;
        current_SSE  = 0;
        for(unsigned int i = 0; i < n_clusters; i++)
        {
            current_SSE += SSE[i];
        }

        if(++n_iter > max_iter)
        {
            delete [] SSE;
            return label;
        }
    }

    delete [] SSE;
    return label;
}