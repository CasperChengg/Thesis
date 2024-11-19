#include <ctime> // timespec, clock_gettime
#include "../../../inc/decision_tree_classifier.h"
#include "../../../inc/file_operations.h"
#include "../../../inc/validation.h"

typedef struct MultiTestMetrics{
    std::vector<float> precision;
    std::vector<float> recall;
    std::vector<float> f1_score;
    std::vector<float> g_mean;
    std::vector<float> elapsed_time_ms;
}MultiTestMetrics;

float GetMultiTestAverage(const std::vector<float> &multi_test_score)
{
    float total_score = 0.f;
    for(const float score : multi_test_score){
        total_score += score;
    }

    return total_score / multi_test_score.size();
}

int main(int argc, char *argv[])
{
    std::string file_path = "../../../dataset/" + (std::string)argv[1] + "-5-fold/" + (std::string)argv[1] + "-5-";
    
    ModelParameters model_parameters = {
        .model_type = MODEL_TYPE,
        .min_samples_split = MIN_SAMPLES_SPLIT,
        .max_purity = MAX_PURITY
    };
    MultiTestMetrics multi_test_metrics = {{}, {}, {}, {}, {}};

    for(uint32_t test_time = 0; test_time < TEST_TIME; test_time++){
        for(uint32_t k = 1; k <= K_FOLD; k++){
            std::string training_path = file_path + std::to_string(k) + "tra.dat";
            std::string testing_path = file_path + std::to_string(k) + "tst.dat";
            Dataset dataset = ReadTrainingAndTestingSet(training_path, testing_path);

            timespec start_ns = {0}, end_ns = {0};
            clock_gettime(CLOCK_MONOTONIC, &start_ns);
            Accuracies accuracies = Validation(dataset.training_set, dataset.testing_set, dataset.n_classes, model_parameters);
            clock_gettime(CLOCK_MONOTONIC, &end_ns);
            float elaped_time_ms = (float)(end_ns.tv_sec - start_ns.tv_sec) * 1000 + 
                                        (float)(end_ns.tv_nsec - start_ns.tv_nsec) / 1000000;

            multi_test_metrics.precision.push_back(accuracies.macro_precision);
            multi_test_metrics.recall.push_back(accuracies.macro_recall);
            multi_test_metrics.f1_score.push_back(accuracies.macro_f1_score);
            multi_test_metrics.g_mean.push_back(accuracies.g_mean);
            multi_test_metrics.elapsed_time_ms.push_back(elaped_time_ms);
        }
    }

    std::cout << GetMultiTestAverage(multi_test_metrics.precision) << std::endl;
    std::cout << GetMultiTestAverage(multi_test_metrics.recall)    << std::endl;
    std::cout << GetMultiTestAverage(multi_test_metrics.f1_score)  << std::endl;
    std::cout << GetMultiTestAverage(multi_test_metrics.g_mean)    << std::endl;
    std::cout << GetMultiTestAverage(multi_test_metrics.elapsed_time_ms)  << std::endl;
}