#include "./minimum_spanning_tree.h"

bool AreAllTrue(bool array[], size_t array_size);

template <class T>
void Prim(std::vector<std::vector<T>> &adjancency_matrix)
{
    size_t num_vertices = adjancency_matrix.size();

    bool *visited = new bool[num_vertices];
    memset(visited, false, num_vertices * sizeof(bool)); // bool is 1-byte long

    T *key = new T[num_vertices];
    for(unsigned int i = 0; i < num_vertices; i++)
    {
        key[i] = std::numeric_limits<T>::max();
    }

    int *parent = new int[num_vertices];
    memset(parent, 0xFF, num_vertices * sizeof(int)); // 0xFFFF = -1

    // set vertec 0 as the first point
    unsigned int newly_added_point = 0;
    parent[0] = 0;
    key[0]    = 0;
    while(!AreAllTrue(visited, num_vertices))
    {
        visited[newly_added_point] = true;
        for(unsigned int neighbor_id = 0; neighbor_id < num_vertices; neighbor_id++)
        {
            if(adjacency_matrix[newly_added_point][neighbor_id] == 0 || visited[neighbor_id] == true)
            {
                continue;
            }
            
            if(adjacency_matrix[newly_added_point][neighbor_id] < key[neighbor_id])
            {
                key[neighbor_id]    = adjacency_matrix[newly_added_point][neighbor_id];
                parent[neighbor_id] = newly_added_point;
            }
        }

        T min_key = std::numeric_limits<T>::max();
        for(unsigned int i = 0; i < num_vertices; i++)
        {   
            if(!visited[i] && key[i] < min_key)
            {
                min_key           = key[i];
                newly_added_point = i;
            }
        }
    }

    for(unsigned int i = 0; i < num_vertices; i++)
    {
        unsigned parent_node = parent[i];
        for(unsigned int neighbor_node = 0; neighbor_node < num_vertices; neighbor_node++)
        {
            if(neighbor_node != parent_node)
            {
                adjacency_matrix[i][neighbor_node] = 0;
            }
        }
    }
}

bool AreAllTrue(bool array[], size_t array_size)
{
    unsigned int i = 0;
    for(; (i < array_size) && array[i]; i++);
    if(i == array_size)
    {
        return true;
    }
    else
    {
        return false;
    }
}