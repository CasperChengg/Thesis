#ifndef DECISION_TREE_H
#define DECISION_TREE_H

// #define SHOW_TREE

#include <cmath>
#include <vector>
#include <cstring> // memset
#include <iostream>
#include <algorithm>

typedef struct SplitPoint
{
    uint32_t feature;
    float value;
    float score;
}SplitPoint;

typedef struct TreeNode
{
    uint32_t label;
    SplitPoint split_point;
    struct TreeNode *right_child;
    struct TreeNode *left_child;
}TreeNode;

TreeNode* CreateDecisionTree(std::vector<std::vector<float>> &training_set, uint32_t n_classes, uint32_t min_samples_split);
uint32_t PredictByDecisionTree(TreeNode *root, std::vector<float> &testing_sample);
#endif