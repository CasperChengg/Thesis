#ifndef BASIC_STRUCTURES_H
#define BASIC_STRUCTURES_H

#include <vector>

typedef struct Dataset{
    size_t dimension;
    size_t num_classes;
    size_t label_index;
    size_t *training_set_class_counts;
    size_t *testing_set_class_counts;
    std::vector<std::vector<float>> training_set;
    std::vector<std::vector<float>> testing_set;
}Dataset;

#endif