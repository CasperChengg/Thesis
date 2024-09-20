#include "./validation.h"

#define TP 0
#define FP 1
#define FN 2

Accuracies Validation(Dataset *dataset, std::string model_type, size_t eta, float pi)
{
    Accuracies accuracies;
    size_t **confusion_matrix = (size_t**)calloc(dataset->num_classes + 1, sizeof(size_t*));
    if(confusion_matrix == NULL)
    {
        printf("./%s:%d: \033[31merror\033[0m: memory allocation error\n", __FILE__, __LINE__);
        exit(1);
    }

    for(size_t i = 1; i <= dataset->num_classes; i++)
    {
        // {TP, FP, FN} for each class
        confusion_matrix[i] = (size_t*)calloc(3, sizeof(size_t));
        if(confusion_matrix[i] == NULL)
        {
            printf("./%s:%d: \033[31merror\033[0m: memory allocation error\n", __FILE__, __LINE__);
            exit(1);
        }
    }
    
    if(model_type == "decision_tree")
    {
        TreeNode *root = CreateDecisionTree(dataset, eta, pi);
        for(size_t i = 0; i < (dataset->testing_set).size(); i++)
        {
            size_t label     = (dataset->testing_set)[i][dataset->label_index]; // ground truth 
            size_t label_hat = PredictByDecisionTree(root, (dataset->testing_set)[i]); // label_hat = model.predict();
            
            if(label == label_hat)
            {
                confusion_matrix[label_hat][TP]++;
            }
            else
            {
                confusion_matrix[label_hat][FP]++;
                confusion_matrix[label][FN]++;
            }
        }
    }
    
    accuracies.precision = 0;
    accuracies.recall    = 0;
    accuracies.f1_score  = 0;
    accuracies.g_mean    = 1;
    
    // In case there are no samples with a specific class in the testing set.
    size_t num_non_zero_classes_in_testing_set = 0; 
    for(size_t i = 1; i <= dataset->num_classes; i++)
    {
        if((dataset->testing_set_class_counts[i] == 0))
        {
            // std::cout << "[empty class]" <<  std::endl;
            continue;
        }
        num_non_zero_classes_in_testing_set++;
        
        float precision = 0;
        if(confusion_matrix[i][TP] + confusion_matrix[i][FN] > 0)
        {
            precision = (float)confusion_matrix[i][TP] / (float)(confusion_matrix[i][TP] + confusion_matrix[i][FN]);
        }

        float recall = 0;    
        if(confusion_matrix[i][TP] + confusion_matrix[i][FP] > 0)
        {
            recall = (float)confusion_matrix[i][TP] / (float)(confusion_matrix[i][TP] + confusion_matrix[i][FP]);
        }

        float f1_score = 0;
        if(precision + recall > 0)
        {
            f1_score = 2 * precision * recall / (precision + recall);
        }

        accuracies.precision += precision;
        accuracies.recall    += recall;
        accuracies.f1_score  += f1_score;
        accuracies.g_mean    *= recall;
        // std::cout << "[" << precision << "," << recall << "," << f1_score << "]" <<  std::endl;
    }
    // std::cout << std::endl;
    accuracies.precision /= num_non_zero_classes_in_testing_set;
    accuracies.recall    /= num_non_zero_classes_in_testing_set;
    accuracies.f1_score  /= num_non_zero_classes_in_testing_set;
    accuracies.g_mean = pow(accuracies.g_mean, 1.0 / num_non_zero_classes_in_testing_set);
    return accuracies;
}