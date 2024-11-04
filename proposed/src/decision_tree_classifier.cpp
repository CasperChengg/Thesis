#include "../inc/decision_tree_classifier.h"

#pragma region FUNCTION_DECLARTION
static void FindBestSplitPoint(TreeNode *node, std::vector<std::vector<float>> &dataset, bool is_exsisting_data[], uint32_t n_classess, uint32_t min_samples_split);
static SplitPoint EvaluateSplitPoint(std::vector<std::vector<float>> &dataset, bool is_existing_data[], uint32_t n_classes, uint32_t feature_idx);
static float CalculateGini(std::vector<std::vector<float>> &dataset, uint32_t n_classes, bool is_existing_data_y[], bool is_existing_data_n[]);
#pragma endregion // FUNCTION_DECLARATION

TreeNode* CreateDecisionTree(std::vector<std::vector<float>> &dataset, uint32_t n_classes, uint32_t min_samples_split)
{
    TreeNode *root;
    try
    {
        root = new TreeNode;
    }
    catch(const std::bad_alloc& error)
    {
        printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());
        exit(1);
    }

    root->left_child = NULL;
    root->right_child = NULL;

    bool *is_existing_data; 
    try
    {
        is_existing_data = new bool[dataset.size()];
    }
    catch(const std::bad_alloc& error)
    {
        printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());
        exit(1);
    }
    memset(is_existing_data, true, dataset.size() * sizeof(bool));

    FindBestSplitPoint(root, dataset, is_existing_data, n_classes, min_samples_split);

    return root;
}
uint32_t PredictByDecisionTree(TreeNode *root, std::vector<float> &testing_sample)
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
static void FindBestSplitPoint(TreeNode *node, std::vector<std::vector<float>> &dataset, bool is_exsisting_data[], uint32_t n_classess, uint32_t min_samples_split)
{
    // Store the number of data points within the partition
    uint32_t n_existing_data = 0;

    // calculate the number of samples of each class within the partition
    uint32_t *class_counts;
    try
    {
        class_counts = new uint32_t[n_classess + 1];
        memset(class_counts, 0, (n_classess + 1) * sizeof(uint32_t));
    }
    catch(const std::bad_alloc &error)
    {
        printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
        exit(1);
    }
     
    for(uint32_t data_idx = 0, data_label_idx = dataset[0].size() - 1; data_idx < dataset.size(); data_idx++)
    {
        if(!is_exsisting_data[data_idx])
        {
            continue;
        }
        uint32_t data_label = dataset[data_idx][data_label_idx];
        class_counts[data_label]++;
    }
    
    #ifdef SHOW_TREE
        std::cout << "--------------------------" << std::endl;
        std::cout << "[";
    #endif //SHOW_TREE
    // Find the majority class within the partition
    uint32_t majority_class_idx = 1;
    for(uint32_t class_idx = 1; class_idx <= n_classess; class_idx++)
    {
        n_existing_data += class_counts[class_idx];
        if(class_counts[majority_class_idx] < class_counts[class_idx])
        {
            majority_class_idx = class_idx;
        }
        #ifdef SHOW_TREE
            std::cout << class_counts[class_idx] << " ";
        #endif //SHOW_TREE
    } 
    #ifdef SHOW_TREE
        std::cout << "]" << std::endl;
    #endif //SHOW_TREE
    // Check the stopping condition
    if(n_existing_data <= min_samples_split || ((float)class_counts[majority_class_idx] / n_existing_data) >= 0.95)
    {
        node->label = majority_class_idx;
        delete []class_counts;
        return;
    }
    delete []class_counts;

    // Initializa best split point
    node->split_point.feature = 0; 
    node->split_point.value = 0; 
    node->split_point.score = 1.1;
    
    // Find the best split point
    uint32_t n_dimensions = dataset[0].size() - 1; // Last column stores the label of the data points
    for(uint32_t dim_idx = 0; dim_idx < n_dimensions; dim_idx++)
    {
        // Find the best split point of the feature
        SplitPoint local_best_split_point = EvaluateSplitPoint(dataset, is_exsisting_data, n_classess, dim_idx);
        if(node->split_point.score >= local_best_split_point.score)
        {
            node->split_point.score   = local_best_split_point.score;
            node->split_point.value   = local_best_split_point.value;
            node->split_point.feature = local_best_split_point.feature;
        }
    }
    #ifdef SHOW_TREE
        std::cout << node->split_point.feature << "<=" << node->split_point.value << ", class: " << majority_class_idx << std::endl;
    #endif //SHOW_TREE
    // partition data with best split point
    uint32_t n_existing_data_y = 0;
    bool *is_existing_data_y;
    
    uint32_t n_existing_data_n = 0;
    bool *is_existing_data_n;
    try
    {
        is_existing_data_y = new bool[dataset.size()];
        memset(is_existing_data_y, false, dataset.size() * sizeof(bool));

        is_existing_data_n = new bool[dataset.size()];
        memset(is_existing_data_n, false, dataset.size() * sizeof(bool));
    }
    catch(const std::bad_alloc &error)
    {
        printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
        exit(1);
    }
    
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++)
    {
        if(!is_exsisting_data[data_idx])
        {
            continue;
        }
        if(dataset[data_idx][node->split_point.feature] <= node->split_point.value)
        {
            is_existing_data_y[data_idx] = true;
            n_existing_data_y++;
        }
        else
        {
            is_existing_data_n[data_idx] = true;
            n_existing_data_n++;
        }
    }

    if(n_existing_data_y > 0)
    {
        try
        {
            node->left_child = new TreeNode;
        }
        catch(const std::bad_alloc &error)
        {
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
            exit(1);
        }
        node->left_child->left_child = NULL;
        node->left_child->right_child = NULL;
        FindBestSplitPoint(node->left_child, dataset, is_existing_data_y, n_classess, min_samples_split);
    }

    if(n_existing_data_n > 0)
    {
        try
        {
            node->right_child = new TreeNode;
        }
        catch(const std::bad_alloc &error)
        {
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
            exit(1);
        }
        node->right_child->left_child = NULL;
        node->right_child->right_child = NULL;
        FindBestSplitPoint(node->right_child, dataset, is_existing_data_n, n_classess, min_samples_split);
    }
}
static SplitPoint EvaluateSplitPoint(std::vector<std::vector<float>> &dataset, bool is_existing_data[], uint32_t n_classes, uint32_t feature_idx)
{
    // Tally the number of data in the partition
    uint32_t n_existing_data = 0;
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++)
    {
        // Count only the existing data
        if(is_existing_data[data_idx])
        {
            n_existing_data++;
        }
    }

    std::vector<float> selected_feature(n_existing_data, 0);
    for(uint32_t data_idx = 0, existing_data_idx = 0; data_idx < dataset.size(); data_idx++)
    {
        if(!is_existing_data[data_idx])
        {
            continue;
        }
        selected_feature[existing_data_idx++] = dataset[data_idx][feature_idx];
    }
    sort(selected_feature.begin(), selected_feature.end());

    SplitPoint best;
    best.value = 0; 
    best.score = 1.1; // gini = [0 1]
    best.feature = feature_idx;

    // Try all possible split points
    for(uint32_t sorted_data_idx = 1; sorted_data_idx < n_existing_data; sorted_data_idx++)
    {
        if(selected_feature[sorted_data_idx - 1] == selected_feature[sorted_data_idx])
        {
            continue;
        }
        
        float mid_point = (selected_feature[sorted_data_idx - 1] + selected_feature[sorted_data_idx]) / 2;
        bool *is_existing_data_y, *is_existing_data_n;
        try
        {
            is_existing_data_y = new bool[dataset.size()];
            memset(is_existing_data_y, false, dataset.size() * sizeof(bool));

            is_existing_data_n = new bool[dataset.size()];
            memset(is_existing_data_n, false, dataset.size() * sizeof(bool));
        }
        catch(const std::bad_alloc &error)
        {
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
            exit(1);
        }

        for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++)
        {
            if(!is_existing_data[data_idx])
            {
                continue;
            }

            if(dataset[data_idx][feature_idx] <= mid_point)
            {
                is_existing_data_y[data_idx] = true;
            }
            else
            {
                is_existing_data_n[data_idx] = true;
            }
        }

        float score = CalculateGini(dataset, n_classes, is_existing_data_y, is_existing_data_n);
        if(score <= best.score)
        {
            best.score = score;
            best.value = mid_point;
        }

        delete[] is_existing_data_y;
        delete[] is_existing_data_n;
    }
    return best;
}
static float CalculateGini(std::vector<std::vector<float>> &dataset, uint32_t n_classes, bool is_existing_data_y[], bool is_existing_data_n[])
{   
    uint32_t *class_counts_y, *class_counts_n;
    try
    {
        class_counts_y = new uint32_t[n_classes + 1];
        memset(class_counts_y, 0, (n_classes + 1) * sizeof(uint32_t));

        class_counts_n = new uint32_t[n_classes + 1];
        memset(class_counts_n, 0, (n_classes + 1) * sizeof(uint32_t));
    }
    catch(const std::bad_alloc &error)
    {
        printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
        exit(1);
    }
    
    uint32_t data_label_idx = dataset[0].size() - 1;
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++)
    {
        uint32_t data_label = dataset[data_idx][data_label_idx];
        
        if(is_existing_data_y[data_idx])
        {
            class_counts_y[data_label]++;
        }
        else if(is_existing_data_n[data_idx])
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
    if(n_existing_data_y > 0)
    {
        for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++)
        {
           float class_ratio_y = (float)class_counts_y[class_idx] / n_existing_data_y;
            gini_y -= class_ratio_y * class_ratio_y;   
        } 
    }
    if(n_existing_data_n > 0)
    {
        for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++)
        {
            float class_ratio_n = (float)class_counts_n[class_idx] / n_existing_data_n;
            gini_n -= class_ratio_n * class_ratio_n;
        }
        
    }
    
    delete[] class_counts_y;
    delete[] class_counts_n;

    // return weighted gini index of the split point
    return ((float)n_existing_data_y / (n_existing_data_y + n_existing_data_n)) * gini_y + ((float)n_existing_data_n / (n_existing_data_y + n_existing_data_n)) * gini_n;
}



