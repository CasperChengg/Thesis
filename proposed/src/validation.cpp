#include "../inc/validation.h"

#define TP 0
#define FP 1
#define FN 2

#pragma region FUNCTION_DECLARATION
static void CalculateClassCounts(std::vector<std::vector<float>> &dataset, uint32_t n_classes, uint32_t class_counts[]);
static Accuracies CalculateAccuracies(std::vector<std::vector<float>> &dataset, TreeNode *root, uint32_t n_classes);
#pragma endregion

Accuracies Validation(Dataset &dataset, std::string model_type, uint32_t min_samples_split)
{
    Accuracies accuracies;
    if(model_type == "decision_tree")
    {
        // Train the decision tree classifier
        TreeNode *root = CreateDecisionTree(dataset.training_set, dataset.n_classes, min_samples_split);
        accuracies = CalculateAccuracies(dataset.testing_set, root, dataset.n_classes);
    }

    return accuracies;
}
static void CalculateClassCounts(std::vector<std::vector<float>> &dataset, uint32_t n_classes, uint32_t class_counts[])
{
    if(class_counts == NULL)
    {
        printf("./%s:%d: error: null pointer\n", __FILE__, __LINE__);
        exit(1);
    }

    uint32_t data_label_idx = dataset[0].size() - 1;

    memset(class_counts, 0, (n_classes + 1) * sizeof(uint32_t));
    for(uint32_t data_idx = 0; data_idx < dataset.size(); data_idx++)
    {   
        uint32_t data_label = dataset[data_idx][data_label_idx];
        class_counts[data_label]++;
    }
}
static Accuracies CalculateAccuracies(std::vector<std::vector<float>> &dataset, TreeNode *root, uint32_t n_classes)
{
    // Allocate memory for 2D confusion matrix
    uint32_t **confusion_matrix;
    try
    {
        // index 0 store nothing
        confusion_matrix = new uint32_t*[n_classes + 1];
        memset(confusion_matrix, 0, (n_classes + 1) * sizeof(uint32_t));
    }
    catch(const std::bad_alloc &error)
    {
        printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
        exit(1);
    }

    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++)
    {
        try
        {
            // {TP, FP, FN} for each class
            confusion_matrix[class_idx] = new uint32_t[3];
            memset(confusion_matrix[class_idx], 0, 3 * sizeof(uint32_t));
        }
        catch(const std::bad_alloc &error)
        {
            printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
            exit(1);
        }
    }

    // Validate the pretrained decision tree classifier using the testing set
    uint32_t data_label_idx = dataset[0].size() - 1;
    for(uint32_t testing_data_idx = 0; testing_data_idx < dataset.size(); testing_data_idx++)
    {
        uint32_t data_label      = dataset[testing_data_idx][data_label_idx]; // ground truth 
        uint32_t predicted_label = PredictByDecisionTree(root, dataset[testing_data_idx]);  // predicted result
        
        if(predicted_label == data_label)
        {
            confusion_matrix[predicted_label][TP]++;
        }
        else
        {
            confusion_matrix[predicted_label][FP]++;
            confusion_matrix[data_label][FN]++;
        }
    }

    // Tally the number of samples per class
    uint32_t *class_counts;
    try
    {
        class_counts = new uint32_t[n_classes + 1];
    }
    catch(const std::bad_alloc &error)
    {
        printf("./%s:%d: error: %s\n", __FILE__, __LINE__, error.what());;
        exit(1);
    }
    CalculateClassCounts(dataset, n_classes, class_counts);

    // Calculate accuracies
    uint32_t n_non_zero_classes = 0;
    Accuracies accuracies= {0.f, 0.f, 0.f, 1.f};
    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++)
    {
        // In case there are no samples with a specific class in the testing set.
        if(class_counts[class_idx] == 0)
        {
            continue;
        }
        n_non_zero_classes++;
        
        float precision = 0;
        if(confusion_matrix[class_idx][TP] + confusion_matrix[class_idx][FN] > 0)
        {
            precision = (float)confusion_matrix[class_idx][TP] / (float)(confusion_matrix[class_idx][TP] + confusion_matrix[class_idx][FN]);
        }

        float recall = 0;    
        if(confusion_matrix[class_idx][TP] + confusion_matrix[class_idx][FP] > 0)
        {
            recall = (float)confusion_matrix[class_idx][TP] / (float)(confusion_matrix[class_idx][TP] + confusion_matrix[class_idx][FP]);
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
    }

    // Get average accuracies
    accuracies.precision /= n_non_zero_classes;
    accuracies.recall    /= n_non_zero_classes;
    accuracies.f1_score  /= n_non_zero_classes;
    accuracies.g_mean     = pow(accuracies.g_mean, 1.f / n_non_zero_classes);

    for(uint32_t class_idx = 1; class_idx <= n_classes; class_idx++) delete []confusion_matrix[class_idx];
    delete []confusion_matrix;
    delete []class_counts;

    return accuracies;
}
