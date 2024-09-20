#ifndef VALIDATION_H
#define VALIDATION_H

#include <cmath>
#include <string>
#include "basic_structures.h"
#include "decision_tree_classifier.h"

typedef struct Accuracies{
    float precision;
    float recall;
    float f1_score;
    float g_mean;
}Accuracies;

Accuracies Validation(Dataset *dataset, std::string model_type, size_t eta, float pi);

#endif
