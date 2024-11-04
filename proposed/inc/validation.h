#ifndef VALIDATION_H
#define VALIDATION_H

#include <cmath>  // pow
#include <string> // memset
#include "../inc/file_operations.h"
#include "../inc/decision_tree_classifier.h"

typedef struct Accuracies{
    float precision;
    float recall;
    float f1_score;
    float g_mean;
}Accuracies;

Accuracies Validation(Dataset &dataset, std::string model_type, uint32_t min_samples_split);

#endif
