#include "../inc/prim.h"

bool AreAllTrue(bool array[], size_t array_size);

template <class T>
std::vector<std::vector<T>> Prim(std::vector<std::vector<T>> &adjacency_matrix)
{
    size_t n_vertices = adjacency_matrix.size();

    bool *is_visited = new bool[n_vertices];
    memset(is_visited, false, n_vertices * sizeof(bool)); // bool is 1-byte long

    // minimum distance from current mst to vertice
    T *key = new T[n_vertices];
    for(unsigned int vertex_idx = 0; vertex_idx < n_vertices; vertex_idx++)
    {
        key[vertex_idx] = std::numeric_limits<T>::max();
    }

    unsigned int *parent = new unsigned int[n_vertices];
    memset(parent, 0xFF, n_vertices * sizeof(int)); // 0xFFFF = -1

    // set vertex 0 as the first point
    unsigned int newly_added_point = 0;
    parent[0] = 0;
    key[0]    = 0;
    while(!AreAllTrue(is_visited, n_vertices)) // loop until all vertices are contained in the MST
    {
        is_visited[newly_added_point] = true;

        // Check neighboring vertices of the vertex newly added to the current MST
        for(unsigned int vertex_idx = 0; vertex_idx < n_vertices; vertex_idx++)
        {
            if(is_visited[vertex_idx])
            {
                continue;
            }
            
            // Update the nearest distance from unvisited vertices to current the MST after a new vertex is added
            T edge_weight = adjacency_matrix[newly_added_point][vertex_idx];
            if((edge_weight) && (edge_weight < key[vertex_idx]))
            {
                key[vertex_idx]    = edge_weight;
                // Record the parent vertex, which is the nearest vertex in the current MST to the unvisited vertex
                parent[vertex_idx] = newly_added_point;
            }
        }

        // Find the nearest unvisited vertex to the current MST and add it to the current MST
        T min_key = std::numeric_limits<T>::max();
        for(unsigned int vertex_idx = 0; vertex_idx < n_vertices; vertex_idx++)
        {   
            if((!is_visited[vertex_idx]) && key[vertex_idx] < min_key)
            {
                min_key           = key[vertex_idx];
                newly_added_point = vertex_idx;
            }
        }
    }

    // Generate the adjacency matrix of the MST
    std::vector<std::vector<T>> mst_adjacency_matrix(n_vertices, std::vector<T>(n_vertices, 0));
    for(unsigned int vertex_idx = 0; vertex_idx < n_vertices; vertex_idx++)
    {
        unsigned int parent_vertex = parent[vertex_idx];
        T edge_weight = adjacency_matrix[vertex_idx][parent_vertex];
        mst_adjacency_matrix[vertex_idx][parent_vertex] = edge_weight;
        mst_adjacency_matrix[parent_vertex][vertex_idx] = edge_weight;
    }

    return mst_adjacency_matrix;
}

bool AreAllTrue(bool array[], size_t array_size)
{
    unsigned int idx = 0;
    for(; (idx < array_size) && array[idx]; idx++);
    if(idx == array_size)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//explicit template instantiation
template std::vector<std::vector<float>> Prim(std::vector<std::vector<float>> &);

