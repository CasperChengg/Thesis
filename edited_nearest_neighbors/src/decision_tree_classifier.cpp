#include "decision_tree_classifier.h"

float CalculateGini(Dataset *dataset, std::vector<size_t> partition_sample_index_y, std::vector<size_t>partition_sample_index_n)
{   
    size_t partition_size_y = partition_sample_index_y.size();
    size_t partition_size_n = partition_sample_index_n.size();
    float gini_y = 1.0, gini_n = 1.0;
    
    size_t *class_count_y = (size_t*)malloc((dataset->num_classes + 1) * sizeof(size_t));
    size_t *class_count_n = (size_t*)malloc((dataset->num_classes + 1) * sizeof(size_t));
    for(size_t i = 0; i < (dataset->num_classes + 1); i++)
    {
        class_count_y[i] = 0;
        class_count_n[i] = 0;
    }

#pragma region COUNT_NUM_SAMPLES_PER_CLASS
    for(size_t i = 0; i < partition_size_y; i++)
    {
        size_t label = (dataset->training_set)[partition_sample_index_y[i]][dataset->label_index];
        class_count_y[label]++;
    }
    for(size_t i = 0; i < partition_size_n; i++)
    {
        size_t label = (dataset->training_set)[partition_sample_index_n[i]][dataset->label_index];
        class_count_n[label]++;
    }
#pragma endregion

#pragma region CALCULATE_GINI
    for(size_t i = 1; i <= dataset->num_classes; i++)
    {
        gini_y -= pow(((float)class_count_y[i]/partition_size_y), 2);
    }
    
    for(size_t i = 1; i <= dataset->num_classes; i++)
    {
        gini_n -= pow(((float)class_count_n[i]/partition_size_n), 2);
    }
#pragma endregion

    free(class_count_y);
    free(class_count_n);
    
    // return weighted gini index of the split point
    return ((float)partition_size_y / (partition_size_y + partition_size_n)) * gini_y + ((float)partition_size_n / (partition_size_y + partition_size_n)) * gini_n;
}

SplitPoint EvaluateSplitPoint(Dataset *dataset, std::vector<size_t> partition_sample_index, size_t feature_id)
{
    size_t partition_size = partition_sample_index.size();

#pragma region SORT_SELECTED_FEATURE
    std::vector<float> selected_feature;
    for(size_t i = 0; i < partition_size; i++)
    {
        selected_feature.push_back((dataset->training_set)[partition_sample_index[i]][feature_id]);
    }
    sort(selected_feature.begin(), selected_feature.end());
#pragma endregion

#pragma region TRY_ALL_POSSIBLE_SPLIT_POINTS
    SplitPoint best;
    best.score = 1.1; 
    best.value = 0; 
    best.feature = feature_id; 
    for(size_t i = 1; i < partition_size; i++)
    {
        if(selected_feature[i - 1] == selected_feature[i])
        {
            continue;
        }
        
        float mid = (selected_feature[i - 1] + selected_feature[i]) / 2;

        std::vector<size_t> partition_sample_index_y, partition_sample_index_n;
        for(size_t j = 0; j < partition_size; j++)
        {
            if((dataset->training_set)[partition_sample_index[j]][feature_id] <= mid)
            {
                partition_sample_index_y.push_back(partition_sample_index[j]);
            }
            else
            {
                partition_sample_index_n.push_back(partition_sample_index[j]);
            }
        }
        float score = CalculateGini(dataset, partition_sample_index_y, partition_sample_index_n);
        if(score < best.score)
        {
            best.score = score;
            best.value = mid;
        }
    }
#pragma endregion

    return best;
}

void FindBestSplitPoint(TreeNode *node, Dataset* dataset, std::vector<size_t> partition_samples_index, size_t eta, float pi)
{
#ifdef DEBUG
    std::cout << "================ new node ====================" << std::endl;
#endif

    size_t partition_size = partition_samples_index.size();
    size_t *class_count   = (size_t*)malloc((dataset->num_classes + 1) * sizeof(size_t));
    if(class_count == NULL)
    {
        printf("./%s:%d: \033[31merror\033[0m: memory allocation error\n", __FILE__, __LINE__);
        exit(1);
    }
    for(size_t i = 0; i <= dataset->num_classes; i++) class_count[i] = 0;
    
    // calculate the number of each class and find the maximum one
    for(size_t i = 0; i < partition_size; i++)
    {
        size_t label = (dataset->training_set)[partition_samples_index[i]][dataset->label_index];
        class_count[label]++;
    }

    size_t majority_class = 1;
    for(size_t i = 2; i <= dataset->num_classes; i++)
    {
        if(class_count[majority_class] < class_count[i])
        {
            majority_class = i;
        }
    }

#ifdef DEBUG
    std::cout << partition_size << ", [ ";
    for(size_t i = 1; i <= training_set->num_classes; i++)
    {
        std::cout << class_count[i] << " ";
    }
    std::cout << "]" << std::endl;
#endif
    
    if(partition_size <= eta || ((float)class_count[majority_class] / partition_size) >= pi)
    {
        node->label = majority_class;
        free(class_count);
#ifdef DEBUG
        std::cout << "label = " << node->label << std::endl;
#endif
        return;
    }
    free(class_count);

    // initializa best split point
    node->split_point.feature = 0; 
    node->split_point.value = 0; 
    node->split_point.score = 1.1;
    
    // find the best split point
    for(size_t i = 0; i < dataset->dimension; i++)
    {
        SplitPoint feature_best_split_point = EvaluateSplitPoint(dataset, partition_samples_index, i);
        if(node->split_point.score > feature_best_split_point.score)
        {
            node->split_point.score   = feature_best_split_point.score;
            node->split_point.value   = feature_best_split_point.value;
            node->split_point.feature = feature_best_split_point.feature;
        }
    }

#ifdef DEBUG
    std::cout << node->split_point.feature  << "<=" << node->split_point.value << "," << node->split_point.score << std::endl;
#endif
    
    // partition data with best split point
    std::vector<size_t> partition_samples_index_y, partition_samples_index_n;
    for(size_t i = 0; i < partition_size; i++)
    {
        if((dataset->training_set)[partition_samples_index[i]][node->split_point.feature] <= node->split_point.value)
        {
            partition_samples_index_y.push_back(partition_samples_index[i]);
        }
        else
        {
            partition_samples_index_n.push_back(partition_samples_index[i]);
        }
    }

    if(partition_samples_index_y.size() > 0)
    {
        node->left_child = (TreeNode*)malloc(sizeof(TreeNode));
        if(node->left_child == NULL)
        {
            printf("./%s:%d: \033[31merror\033[0m: memory allocation error\n", __FILE__, __LINE__);;
            exit(1);
        }
        
        node->left_child->left_child = NULL;
        node->left_child->right_child = NULL;
        FindBestSplitPoint(node->left_child, dataset, partition_samples_index_y, eta, pi);
    }

    if(partition_samples_index_n.size() > 0)
    {
        node->right_child = (TreeNode*)malloc(sizeof(TreeNode));
        if(node->right_child == NULL)
        {
            fprintf(stderr, "./%s:%d: \033[31merror\033[0m: memory allocation error\n", __FILE__, __LINE__);
            exit(1);
        }
        node->right_child->left_child = NULL;
        node->right_child->right_child = NULL;
        FindBestSplitPoint(node->right_child, dataset, partition_samples_index_n, eta, pi);
    }
}

TreeNode* CreateDecisionTree(Dataset *dataset, size_t eta, float pi)
{
    TreeNode *root = (TreeNode*)calloc(1, sizeof(TreeNode));
    if(root == NULL)
    {
        printf("./%s:%d: \033[31merror\033[0m: memory allocation error\n", __FILE__, __LINE__);
        exit(1);
    }

    root->left_child = NULL;
    root->right_child = NULL;

    std::vector<size_t> partition_samples_index;
    for(size_t i = 0; i < (dataset->training_set).size(); i++)
    {
        partition_samples_index.push_back(i);
    }
    FindBestSplitPoint(root, dataset, partition_samples_index, eta, pi);

    return root;
}

size_t PredictByDecisionTree(TreeNode *root, std::vector<float> testing_sample)
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

