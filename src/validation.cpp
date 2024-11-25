#include "../inc/validation.h"

// #define TP 0
// #define FP 1
// #define FN 2
// #define TN 3

static Accuracies CalcAccForDecisionTree(const std::vector<std::vector<float>> &testing_set, 
                                            const uint32_t n_training_classes, 
                                                TreeNode *root)
{
    // The number of classes in the testing set may be smaller than in the training set
    // n_testing_classes <= n_training_classes
    uint32_t n_testing_classes = 0;
    const uint32_t data_label_idx = testing_set[0].size() - 1;
     
    Accuracies accuracies = {
        .macro_precision = 0.f,
        .macro_recall    = 0.f,
        .macro_f1_score  = 0.f,
        .g_mean          = 1.f,
        .FDR             = {},
        .FOR             = {}
    };
    
    std::vector<std::vector<uint32_t>> confusion_matrix(n_training_classes + 1, std::vector<uint32_t>(n_training_classes + 1, 0));

    for(uint32_t testing_data_idx = 0; testing_data_idx < testing_set.size(); testing_data_idx++){
        uint32_t data_label      = testing_set[testing_data_idx][data_label_idx];               // Ground truth
        uint32_t predicted_label = PredictByDecisionTree(root, testing_set[testing_data_idx]);  // Prediction
        confusion_matrix[predicted_label][data_label]++;
        // if(predicted_label == data_label) {
        //     confusion_matrix[predicted_label][data_label]++;
        // }
        // else{
        //     confusion_matrix[predicted_label][FP]++;
        //     confusion_matrix[data_label][FN]++;
        // }
    }

    // Read confusion matrix
    for(uint32_t class_idx = 1; class_idx <= n_training_classes; class_idx++){ // Class labels start from 1
        // In case there are no samples with a specific class in the testing set.
        uint32_t class_count = 0;
        uint32_t TP = 0, FP = 0, FN = 0, TN = 0;
        TP = confusion_matrix[class_idx][class_idx];
        for(uint32_t col_idx = 1; col_idx <= n_training_classes; col_idx++){
            if(col_idx != class_idx){
                FP += confusion_matrix[class_idx][col_idx];
                TN += confusion_matrix[col_idx][col_idx];
            }
        }

        for(uint32_t row_idx = 1; row_idx <= n_training_classes; row_idx++){
            if(row_idx != class_idx){
                FN += confusion_matrix[row_idx][class_idx];
            }
        }

        if((TP + FN) > 0){
            float precision = 0.f;
            if((confusion_matrix[class_idx][TP] + confusion_matrix[class_idx][FN]) > 0){
                precision = (float)confusion_matrix[class_idx][TP] / 
                                (float)(confusion_matrix[class_idx][TP] + confusion_matrix[class_idx][FN]);
            }

            float recall = 0.f;    
            if((confusion_matrix[class_idx][TP] + confusion_matrix[class_idx][FP]) > 0){
                recall = (float)confusion_matrix[class_idx][TP] / 
                            (float)(confusion_matrix[class_idx][TP] + confusion_matrix[class_idx][FP]);
            }

            float f1_score = 0.f;
            if((precision + recall) > 0){
                f1_score = 2 * precision * recall / (precision + recall);
            }

            float FDR = 0.f;
            if((confusion_matrix[class_idx][FP] + confusion_matrix[class_idx][TP]) > 0){
                FDR = (float)confusion_matrix[class_idx][FP] / 
                        (float)(confusion_matrix[class_idx][FP] + confusion_matrix[class_idx][TP]);
            }

            float FOR = 0.f;
            if((confusion_matrix[class_idx][FN] + confusion_matrix[class_idx][TN]) > 0){
                FOR = (float)confusion_matrix[class_idx][FN] / 
                        (float)(confusion_matrix[class_idx][FN] + confusion_matrix[class_idx][TN]);
            }
            
            accuracies.macro_precision += precision;
            accuracies.macro_recall    += recall;
            accuracies.macro_f1_score  += f1_score;
            accuracies.g_mean          *= recall;
            accuracies.FDR.push_back(FDR);
            accuracies.FOR.push_back(FOR);
        }
    }

    accuracies.macro_precision /= n_testing_classes;
    accuracies.macro_recall    /= n_testing_classes;
    accuracies.macro_f1_score  /= n_testing_classes;
    accuracies.g_mean           = pow(accuracies.g_mean, 1.f / n_testing_classes);

    return accuracies;
}

Accuracies Validation(const std::vector<std::vector<float>> &training_set, 
                        const std::vector<std::vector<float>> &testing_set, 
                            const uint32_t n_training_classes, 
                                const ModelParameters model_parameters)
{
    Accuracies accuracies;
    if(model_parameters.model_type == "decision_tree"){
        TreeNode *root = CreateDecisionTree(training_set, n_training_classes, model_parameters.min_samples_split, model_parameters.max_purity);
        accuracies = CalcAccForDecisionTree(testing_set, n_training_classes, root);
    }
    /**
     * Add other multiclass classifiers here
     */

    return accuracies;
}
