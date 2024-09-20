#ifndef DECISION_TREE_H
#define DECISION_TREE_H

// #define DEBUG

#include <cmath>
#include <iostream>
#include <algorithm>
#include "basic_structures.h"

typedef struct SplitPoint
{
    size_t feature;
    float value;
    float score;
}SplitPoint;

typedef struct TreeNode
{
    size_t label;
    SplitPoint split_point;
    struct TreeNode *right_child;
    struct TreeNode *left_child;
}TreeNode;

TreeNode* CreateDecisionTree(Dataset *training_set, size_t eta, float pi);
size_t PredictByDecisionTree(TreeNode *root, std::vector<float> testing_sample);
#endif