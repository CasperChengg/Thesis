#include <ctime> // timespec, clock_gettime
#include "../../inc/validation.h"
#include "../../inc/file_operations.h"
#include "../inc/proposed.h"

typedef struct MultiTestMetrics{
    std::vector<float> precision;
    std::vector<float> recall;
    std::vector<float> f1_score;
    std::vector<float> g_mean;
    std::vector<float> elapsed_time_ms;
}MultiTestMetrics;

inline float GetKFoldTestAverage(const std::vector<float> &k_fold_test_metric)
{
    return std::accumulate(k_fold_test_metric.begin(), k_fold_test_metric.end(), 0.f) / k_fold_test_metric.size();
}

inline float GetMultiTestAverage(const std::vector<float> &multi_test_metric)
{
    float total_score = std::accumulate(multi_test_metric.begin(), multi_test_metric.end(), 0.f);
    total_score -=  *std::max_element(multi_test_metric.begin(), multi_test_metric.end());
    total_score -=  *std::min_element(multi_test_metric.begin(), multi_test_metric.end()); 
    return total_score / (multi_test_metric.size() - 2);
}

int main(int argc, char *argv[])
{
#ifdef DEBUG
    std::cout << "====================Dataset Processing Report ====================" << std::endl;
#endif //DEBUG
    std::string file_path = "../../datasets/" + (std::string)argv[1] + "-5-fold/" + (std::string)argv[1] + "-5-";

    ModelParameters model_parameters = {
        .model_type = MODEL_TYPE,
        .min_samples_split = MIN_SAMPLES_SPLIT,
        .max_purity = MAX_PURITY
    };
    MultiTestMetrics multi_test_metrics = {{}, {}, {}, {}, {}};

    for(uint32_t test_time = 0; test_time < TEST_TIME; test_time++){
        MultiTestMetrics k_fold_test_metrics = {{}, {}, {}, {}, {}};
        for(uint32_t k = 1; k <= K_FOLD; k++){
            std::string training_path = file_path + std::to_string(k) + "tra.dat";
            std::string testing_path = file_path + std::to_string(k) + "tst.dat";
            Dataset dataset = ReadTrainingAndTestingSet(training_path, testing_path);

            timespec start_ns = {0}, end_ns = {0};
            clock_gettime(CLOCK_MONOTONIC, &start_ns);
            Proposed(dataset.training_set, dataset.n_classes, KNN, model_parameters);
            Accuracies accuracies = Validation(dataset.training_set, dataset.testing_set, dataset.n_classes, model_parameters);
            clock_gettime(CLOCK_MONOTONIC, &end_ns);
            float elaped_time_ms = (float)(end_ns.tv_sec - start_ns.tv_sec) * 1000 + 
                                        (float)(end_ns.tv_nsec - start_ns.tv_nsec) / 1000000;

            k_fold_test_metrics.precision.push_back(accuracies.macro_precision);
            k_fold_test_metrics.recall.push_back(accuracies.macro_recall);
            k_fold_test_metrics.f1_score.push_back(accuracies.macro_f1_score);
            k_fold_test_metrics.g_mean.push_back(accuracies.g_mean);
            k_fold_test_metrics.elapsed_time_ms.push_back(elaped_time_ms);
        }

        multi_test_metrics.precision.push_back(GetKFoldTestAverage(k_fold_test_metrics.precision));
        multi_test_metrics.recall.push_back(GetKFoldTestAverage(k_fold_test_metrics.recall));
        multi_test_metrics.f1_score.push_back(GetKFoldTestAverage(k_fold_test_metrics.f1_score));
        multi_test_metrics.g_mean.push_back(GetKFoldTestAverage(k_fold_test_metrics.g_mean));
        multi_test_metrics.elapsed_time_ms.push_back(GetKFoldTestAverage(k_fold_test_metrics.elapsed_time_ms));
    }
#ifdef DEBUG
    std::cout << "-Testing Result" << std::endl;
    std::cout << "\t" << GetMultiTestAverage(multi_test_metrics.precision) << std::endl;
    std::cout << "\t" << GetMultiTestAverage(multi_test_metrics.recall)    << std::endl;
    std::cout << "\t" << GetMultiTestAverage(multi_test_metrics.f1_score)  << std::endl;
    std::cout << "\t" << GetMultiTestAverage(multi_test_metrics.g_mean)    << std::endl;
    std::cout << "\t" << GetMultiTestAverage(multi_test_metrics.elapsed_time_ms)  << std::endl;
#else
    std::cout << GetMultiTestAverage(multi_test_metrics.precision) << std::endl;
    std::cout << GetMultiTestAverage(multi_test_metrics.recall)    << std::endl;
    std::cout << GetMultiTestAverage(multi_test_metrics.f1_score)  << std::endl;
    std::cout << GetMultiTestAverage(multi_test_metrics.g_mean)    << std::endl;
    std::cout << GetMultiTestAverage(multi_test_metrics.elapsed_time_ms)  << std::endl;
#endif
}