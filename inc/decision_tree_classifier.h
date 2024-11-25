#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <cmath>
#include <vector>
#include <cstring> // memset
#include <iostream>
#include <algorithm>

typedef struct SplitPoint{
    uint32_t feature;
    float value;
    float score;
}SplitPoint;

typedef struct TreeNode{
    uint32_t label;
    SplitPoint split_point;
    struct TreeNode *right_child;
    struct TreeNode *left_child;
}TreeNode;

TreeNode* CreateDecisionTree(const std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const uint32_t min_samples_split, const float max_purity);
uint32_t PredictByDecisionTree(TreeNode *root, const std::vector<float> &testing_sample);
#endif