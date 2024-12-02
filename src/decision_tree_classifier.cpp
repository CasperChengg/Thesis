#include "../inc/decision_tree_classifier.h"

#define IDX 0
#define ATTR 1
#define LABEL 2
static float CalculateGini(const uint32_t partition_size_y, std::vector<uint32_t> &class_counts_y, 
                            const uint32_t partition_size_n, const std::vector<uint32_t> &class_counts_n)
{
    float gini_y = 1.0;
    if(partition_size_y > 0){
        for(uint32_t class_idx = 1; class_idx < class_counts_y.size(); class_idx++){
            float class_ratio = (float)class_counts_y[class_idx] / partition_size_y;
            gini_y -= class_ratio * class_ratio; 
        }
    }
   
    float gini_n = 1.0;
    if(partition_size_n > 0){
        for(uint32_t class_idx = 1; class_idx < class_counts_n.size(); class_idx++){
            float class_ratio = (float)class_counts_n[class_idx] / partition_size_n;
            gini_n -= class_ratio * class_ratio; 
        }
    }

    // std::cout << "[" ;
    // for(uint32_t class_idx = 1; class_idx < class_counts_y.size(); class_idx++){
    //     std::cout << class_counts_y[class_idx] << ", ";
    // }
    // std::cout << "]" << std::endl;
    // std::cout << "[" ;
    // for(uint32_t class_idx = 1; class_idx < class_counts_n.size(); class_idx++){
    //     std::cout << class_counts_n[class_idx] << ", ";
    // }
    // std::cout << "]" << std::endl;
    // std::cout << gini_y << ", " << gini_n << std::endl;

    return (float)partition_size_y / (partition_size_y + partition_size_n) * gini_y +
                                    (float)partition_size_n / (partition_size_y + partition_size_n) * gini_n;
}

static SplitPoint EvaluateSplitPoint(const std::vector<std::vector<std::vector<float>>> &sorted_features, const std::vector<bool> &is_existing_data, const uint32_t n_classes, const uint32_t feature_idx)
{
    SplitPoint best = {feature_idx, 0.f, 1.1};
    uint32_t best_split_left_idx = 0, best_split_right_idx = 0;
    uint32_t split_left_idx = 0, split_right_idx = 0;

    bool update_best_split = false;
    
    uint32_t partition_size_y = 0;
    std::vector<uint32_t> class_counts_y(n_classes + 1, 0);
    std::vector<uint32_t> class_counts_n(n_classes + 1, 0);
    for(uint32_t idx = 0; idx < sorted_features[feature_idx].size(); idx++){
        uint32_t data_idx = sorted_features[feature_idx][idx][IDX];
        if(is_existing_data[data_idx]){
            // std::cout << sorted_features[feature_idx][idx][ATTR] << ", ";
            uint32_t label = sorted_features[feature_idx][idx][LABEL];
            if(partition_size_y == 0){
                partition_size_y++;
                class_counts_y[label]++;
                split_right_idx = idx;
            }
            else{
                class_counts_n[label]++;
            }
            // else{
            //     if(attr != current_split_right){
            //         current_split_
            //         best_split_right_idx = idx;
            //         update_best_split = false;
            //     }
            //     class_counts_n[label]++;
            // }
        }
    }
    // std::cout << std::endl;
    uint32_t partition_size_n = std::accumulate(class_counts_n.begin(), class_counts_n.end(), 0.f);

    float class_ratio = 0.f;
    float best_weighted_gini = 1.1;
    // std::cout << "[" << split_left_idx << ", " << split_right_idx << "]" << sorted_features[feature_idx].size() << std::endl;
    while(split_left_idx < sorted_features[feature_idx].size() && split_right_idx < sorted_features[feature_idx].size()){
        split_left_idx = split_right_idx;
        // std::cout << "[" << split_left_idx << ", " << split_right_idx << "]" << sorted_features[feature_idx].size() << std::endl;
        for(split_right_idx = split_left_idx + 1; split_right_idx < sorted_features[feature_idx].size(); split_right_idx++){
            uint32_t data_idx = sorted_features[feature_idx][split_right_idx][IDX];
            uint32_t label    = sorted_features[feature_idx][split_right_idx][LABEL];
            bool is_diff = sorted_features[feature_idx][split_left_idx][ATTR] != 
                                sorted_features[feature_idx][split_right_idx][ATTR];

            if(is_existing_data[data_idx]){
                if(is_diff){
                    // std::cout << sorted_features[feature_idx][split_left_idx][ATTR] << ", " << sorted_features[feature_idx][split_right_idx][ATTR] << ", " << is_diff<< std::endl;
                    float weighted_gini = CalculateGini(partition_size_y, class_counts_y, partition_size_n, class_counts_n);
                    // std::cout << "gini = " << weighted_gini << std::endl;
                    if(weighted_gini < best_weighted_gini){
                        best_weighted_gini = weighted_gini;
                        best_split_left_idx = split_left_idx;
                        best_split_right_idx = split_right_idx;
                    }
                    // std::cout << "[" ;
                    // for(uint32_t class_idx = 1; class_idx < class_counts_y.size(); class_idx++){
                    //     std::cout << class_counts_y[class_idx] << ", ";
                    // }
                    // std::cout << "]" << std::endl;
                    // std::cout << "[" ;
                    // for(uint32_t class_idx = 1; class_idx < class_counts_n.size(); class_idx++){
                    //     std::cout << class_counts_n[class_idx] << ", ";
                    // }
                    // std::cout << "]" << std::endl;
                    partition_size_y++;
                    class_counts_y[label]++;
                    partition_size_n--;
                    class_counts_n[label]--; 
                    break;
                }
                partition_size_y++;
                class_counts_y[label]++;
                partition_size_n--;
                class_counts_n[label]--;    
            } 
        }
    }
    
    best.score = best_weighted_gini;
    best.value = (sorted_features[feature_idx][best_split_left_idx][ATTR] + 
                    sorted_features[feature_idx][best_split_right_idx][ATTR]) / 2;
    // std::cout << sorted_features[feature_idx][best_split_left_idx][ATTR] << ", " << sorted_features[feature_idx][best_split_right_idx][ATTR] << std::endl;
    return best;
}

static void FindBestSplitPoint(TreeNode *node, const std::vector<std::vector<std::vector<float>>> &sorted_features, std::vector<bool> &is_existing_data, const uint32_t n_classess, const uint32_t min_samples_split, const float max_purity)
{   
    const uint32_t partition_size = std::count_if(is_existing_data.begin(), is_existing_data.end(), 
                                                    [](bool is_existing){return is_existing;});                                            
    const uint32_t n_features = (sorted_features.size());
    std::vector<uint32_t> class_counts((n_classess + 1), 0);
    for(uint32_t idx = 0; idx < sorted_features[0].size(); idx++){
        uint32_t data_idx = sorted_features[0][idx][IDX];
        if(is_existing_data[data_idx]){
            uint32_t label = sorted_features[0][idx][LABEL];
            class_counts[label]++;
        }
    }
    
    auto max_it = std::max_element(class_counts.begin(), class_counts.end());
    const uint32_t majority_label = std::distance(class_counts.begin(), max_it);
    const uint32_t majority_count = *max_it;

    // std::cout << partition_size << ", " << (float)majority_count / partition_size << std::endl;
    if(partition_size <= min_samples_split || (float)majority_count / partition_size >= max_purity){
        node->label = majority_label;
        return;
    }

    node->split_point = {0, 0.f, 1.1};
    
    for(uint32_t feature_idx = 0; feature_idx < n_features; feature_idx++){
        SplitPoint feature_best_split_point = EvaluateSplitPoint(sorted_features, is_existing_data, n_classess, feature_idx);
        // std::cout << feature_best_split_point.feature << "<=" << feature_best_split_point.value << "(" << feature_best_split_point.score << ")" << std::endl;
        if(node->split_point.score >= feature_best_split_point.score){
            memcpy(&(node->split_point), &(feature_best_split_point), sizeof(SplitPoint));
        }
    }
    // std::cout << node->split_point.feature << "<=" << node->split_point.value << "(" << node->split_point.score << ")" << std::endl;
    // std::cout << std::endl;
    
    // partition data with best split point
    bool split_partition_y = false, split_partition_n = false;
    std::vector<bool> is_existing_data_y(is_existing_data.size(), false);
    std::vector<bool> is_existing_data_n(is_existing_data.size(), false);

    for(uint32_t idx = 0; idx < sorted_features[node->split_point.feature].size(); idx++){
        uint32_t data_idx = sorted_features[node->split_point.feature][idx][IDX];
        
        if(is_existing_data[data_idx]){
            float data_attr = sorted_features[node->split_point.feature][idx][ATTR];
            if(data_attr <= node->split_point.value){
                is_existing_data_y[data_idx] = true;
                split_partition_y = true;
            }
            else{
                is_existing_data_n[data_idx] = true;
                split_partition_n = true;
            }
        }
    }

    if(split_partition_y){
        try{
            node->left_child = new TreeNode;
            node->left_child->left_child = NULL;
            node->left_child->right_child = NULL;
        }
        catch(const std::bad_alloc &error){
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
            exit(1);
        }
        FindBestSplitPoint(node->left_child, sorted_features, is_existing_data_y, n_classess, min_samples_split, max_purity);
    }

    // Recursively split the partition n if needed 
    if(split_partition_n){
        try{
            node->right_child = new TreeNode;
            node->right_child->left_child = NULL;
            node->right_child->right_child = NULL;
        }
        catch(const std::bad_alloc &error){
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());
            exit(1);
        }
        FindBestSplitPoint(node->right_child, sorted_features, is_existing_data_n, n_classess, min_samples_split, max_purity);
    }
}

TreeNode* CreateDecisionTree(const std::vector<std::vector<float>> &training_set, const uint32_t n_classes, const uint32_t min_samples_split, const float max_purity)
{
    TreeNode *root;
    try{
        root = new TreeNode;
        root->left_child = NULL;
        root->right_child = NULL;
    }
    catch(const std::bad_alloc& error){
        printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());
        exit(1);
    }

    // dimension * data_size * 3(idx, value, label)
    std::vector<std::vector<std::vector<float>>> sorted_features(training_set[0].size() - 1, std::vector<std::vector<float>>(training_set.size(), std::vector<float>(3, 0.f)));
    
    const uint32_t label_idx = training_set[0].size() - 1;
    for(uint32_t dim_idx = 0; dim_idx < training_set[0].size() - 1; dim_idx++){
        for(uint32_t data_idx = 0; data_idx < training_set.size(); data_idx++){
            sorted_features[dim_idx][data_idx][IDX]   = data_idx;
            sorted_features[dim_idx][data_idx][ATTR]  = training_set[data_idx][dim_idx];
            sorted_features[dim_idx][data_idx][LABEL] = training_set[data_idx][label_idx];
        }
        std::sort(sorted_features[dim_idx].begin(), sorted_features[dim_idx].end(), 
                    [](const std::vector<float> &a, const std::vector<float> &b){return a[ATTR] < b[ATTR];});
    }

    std::vector<bool> is_existing_data(training_set.size(), true);
    FindBestSplitPoint(root, sorted_features, is_existing_data, n_classes, min_samples_split, max_purity);

    return root;
}

uint32_t PredictByDecisionTree(TreeNode *root, const std::vector<float> &testing_sample)
{
    if(root->left_child == NULL && root->right_child == NULL){
        return root->label;
    }

    if(testing_sample[root->split_point.feature] <= root->split_point.value){
        return PredictByDecisionTree(root->left_child, testing_sample);
    }
    else{
        return PredictByDecisionTree(root->right_child, testing_sample);
    }
}






