#ifndef VALIDATION_H
#define VALIDATION_H

#include <cmath>  // pow
#include <vector> // std::vector
#include "../inc/decision_tree_classifier.h" // CreateDecisionTree, PredictByDecisionTree

typedef struct Accuracies{
    float macro_precision;
    float macro_recall;
    float macro_f1_score;
    float g_mean;
}Accuracies;

typedef struct ModelParameters{
    std::string model_type;
    uint32_t min_samples_split; // Parameters for the decision tree
    float max_purity;
}ModelParameters;

Accuracies Validation(const std::vector<std::vector<float>> &training_set, const std::vector<std::vector<float>> &testing_set, const uint32_t n_classes, const ModelParameters model_parameters);

#endif
