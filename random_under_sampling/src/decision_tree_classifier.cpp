#include "../inc/decision_tree_classifier.h"

static float CalculateGini(const std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const std::vector<bool> &is_existing_data_y, const std::vector<bool> &is_existing_data_n)
{   
    std::vector<uint32_t> class_counts_y((n_classes + 1), 0);
    std::vector<uint32_t> class_counts_n((n_classes + 1), 0);
    
    uint32_t data_label_idx = training_set[0].size() - 1;
    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++)
    {
        uint32_t data_label = training_set[training_data_idx][data_label_idx];
        if(is_existing_data_y[training_data_idx])
        {
            class_counts_y[data_label]++;
        }
        else if(is_existing_data_n[training_data_idx])
        {
            class_counts_n[data_label]++;
        }
    }

    uint32_t n_existing_data_y = 0, n_existing_data_n = 0;
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++)
    {
        n_existing_data_y += class_counts_y[class_idx];
        n_existing_data_n += class_counts_n[class_idx];
    }

    float gini_y = 1.0, gini_n = 1.0;
    if(n_existing_data_y > 0){
        for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
            float class_ratio_y = (float)class_counts_y[class_idx] / n_existing_data_y;
            gini_y -= class_ratio_y * class_ratio_y;   
        } 
    }

    if(n_existing_data_n > 0){
        for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++){
            float class_ratio_n = (float)class_counts_n[class_idx] / n_existing_data_n;
            gini_n -= class_ratio_n * class_ratio_n;
        }
        
    }

    // return weighted gini index of the split point
    return ((float)n_existing_data_y / (n_existing_data_y + n_existing_data_n)) * gini_y + ((float)n_existing_data_n / (n_existing_data_y + n_existing_data_n)) * gini_n;
}

static SplitPoint EvaluateSplitPoint(const std::vector<std::vector<float>> &training_set, const std::vector<bool> &is_existing_data, const uint32_t n_classes, const uint32_t feature_idx)
{
    uint32_t n_existing_data = 0;
    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++)
    {
        // Count only the existing data
        if(is_existing_data[training_data_idx])
        {
            n_existing_data++;
        }
    }

    std::vector<float> selected_feature(n_existing_data, 0);
    for(uint32_t training_data_idx = 0, existing_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++)
    {
        if(!is_existing_data[training_data_idx])
        {
            continue;
        }
        selected_feature[existing_data_idx++] = training_set[training_data_idx][feature_idx];
    }
    sort(selected_feature.begin(), selected_feature.end());

    // Try all possible split points
    SplitPoint best = {feature_idx, 0.f, 1.1};
    for(uint32_t sorted_data_idx = 1; sorted_data_idx < n_existing_data; sorted_data_idx++){
        if(selected_feature[sorted_data_idx - 1] == selected_feature[sorted_data_idx]){
            continue;
        }
        
        float mid_point = (selected_feature[sorted_data_idx - 1] + selected_feature[sorted_data_idx]) / 2;
        
        std::vector<bool> is_existing_data_y(training_set.size(), false);
        std::vector<bool> is_existing_data_n(training_set.size(), false);
        for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){
            if(!is_existing_data[training_data_idx]){
                continue;
            }

            if(training_set[training_data_idx][feature_idx] <= mid_point){
                is_existing_data_y[training_data_idx] = true;
            }
            else{
                is_existing_data_n[training_data_idx] = true;
            }
        }

        float score = CalculateGini(training_set, n_classes, is_existing_data_y, is_existing_data_n);
        if(score <= best.score)
    {
            best.score = score;
            best.value = mid_point;
        }
    }
    return best;
}

static void FindBestSplitPoint(TreeNode *node, const std::vector<std::vector<float>> &training_set, std::vector<bool> &is_exsisting_data, const uint32_t n_classess, const uint32_t min_samples_split, const float max_purity)
{
    std::vector<uint32_t> class_counts((n_classess + 1), 0);
    
    for(uint32_t training_data_idx = 0, data_label_idx = (training_set[0].size() - 1); training_data_idx < training_set.size(); training_data_idx++){
        if(!is_exsisting_data[training_data_idx]){
            continue;
        }
        uint32_t data_label = training_set[training_data_idx][data_label_idx];
        class_counts[data_label]++;
    }

    // Find the majority class within the partition
    uint32_t majority_class_idx = 1, n_existing_data = 0;
    for(uint32_t class_idx = 1; class_idx <= n_classess; class_idx++){
        n_existing_data += class_counts[class_idx];
        if(class_counts[majority_class_idx] < class_counts[class_idx]){
            majority_class_idx = class_idx;
        }
    } 

    // Check the stopping condition
    if(n_existing_data <= min_samples_split || ((float)class_counts[majority_class_idx] / n_existing_data) >= max_purity)
    {
        node->label = majority_class_idx;
        return;
    }

    // Initialize the best split point
    node->split_point = {0, 0.f, 1.1};
    
    // Find the best split point
    const uint32_t n_dimensions = (training_set[0].size() - 1);
    for(uint32_t dim_idx = 0; dim_idx < n_dimensions; dim_idx++){
        // Find the best split point of the feature
        SplitPoint local_best_split_point = EvaluateSplitPoint(training_set, is_exsisting_data, n_classess, dim_idx);
        if(node->split_point.score >= local_best_split_point.score){
            memcpy(&(node->split_point), &(local_best_split_point), sizeof(SplitPoint));
        }
    }

    // partition data with best split point
    bool split_partition_y = false;
    std::vector<bool> is_existing_data_y(training_set.size(), false);
    
    bool split_partition_n = false;
    std::vector<bool> is_existing_data_n(training_set.size(), false);
    
    for(uint32_t training_data_idx = 0; training_data_idx < training_set.size(); training_data_idx++){
        if(!is_exsisting_data[training_data_idx]){
            continue;
        }
        if(training_set[training_data_idx][node->split_point.feature] <= node->split_point.value){
            is_existing_data_y[training_data_idx] = true;
            split_partition_y = true;
        }
        else{
            is_existing_data_n[training_data_idx] = true;
            split_partition_n = true;
        }
    }

    // Recursively split the partition y if needed 
    if(split_partition_y)
    {
        try
        {
            node->left_child = new TreeNode;
            node->left_child->left_child = NULL;
            node->left_child->right_child = NULL;
        }
        catch(const std::bad_alloc &error)
        {
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
            exit(1);
        }
        
        FindBestSplitPoint(node->left_child, training_set, is_existing_data_y, n_classess, min_samples_split, max_purity);
    }

    // Recursively split the partition n if needed 
    if(split_partition_n)
    {
        try
        {
            node->right_child = new TreeNode;
            node->right_child->left_child = NULL;
            node->right_child->right_child = NULL;
        }
        catch(const std::bad_alloc &error)
        {
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());
            exit(1);
        }
        FindBestSplitPoint(node->right_child, training_set, is_existing_data_n, n_classess, min_samples_split, max_purity);
    }
}

TreeNode* CreateDecisionTree(const std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const uint32_t min_samples_split, const float max_purity)
{
    TreeNode *root;
    try
    {
        root = new TreeNode;
        root->left_child = NULL;
        root->right_child = NULL;
    }
    catch(const std::bad_alloc& error)
    {
        printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());
        exit(1);
    }

    // Use is_existing_data to indicate if the data exists in the current partition
    std::vector<bool> is_existing_data(training_set.size(), true);

    // Recursively find the best split point for each node, starting from the root
    FindBestSplitPoint(root, training_set, is_existing_data, n_classes, min_samples_split, max_purity);

    return root;
}

uint32_t PredictByDecisionTree(TreeNode *root, const std::vector<float> &testing_sample)
{
    if(root->left_child == NULL && root->right_child == NULL)
    {
        return root->label;
    }

    if(testing_sample[root->split_point.feature] <= root->split_point.value)
    {
        return PredictByDecisionTree(root->left_child, testing_sample);
    }
    else
    {
        return PredictByDecisionTree(root->right_child, testing_sample);
    }
}






