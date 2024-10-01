#include <ctime>
#include "validation.h"
#include "file_operations.h"
#include "../inc/proposed.h"

int main(int argc, char *argv[])
{
    float elaped_time_ms = 0;
    timespec start_ns, end_ns;
    std::vector<Accuracies> multi_test_accuracies;

    std::string file_path = "../../dataset/" + (std::string)argv[1] + "-5-fold/" + (std::string)argv[1] + "-5-";

    for(size_t i = 0; i < TEST_TIME; i++)
    {
        for(int k = 1; k <= K_FOLD; k++)
        {
            std::string dataset_path = file_path + std::to_string(k);
            Dataset dataset = ReadDataset(dataset_path);
            clock_gettime(CLOCK_MONOTONIC, &start_ns);
            Proposed<float>(dataset.training_set, dataset.num_classes, KNN);
            Accuracies accuracies = Validation(&dataset, "decision_tree", ETA, PI);
            clock_gettime(CLOCK_MONOTONIC, &end_ns);

            elaped_time_ms += (float)(end_ns.tv_sec - start_ns.tv_sec) * 1000 + (float)(end_ns.tv_nsec - start_ns.tv_nsec) / 1000000;
            multi_test_accuracies.push_back(accuracies);
            
        }
    }
    
    float avg_precision = 0, avg_recall = 0, avg_f1_score = 0, avg_g_mean = 0, avg_elapsed_time_ms = 0;
    for(size_t i = 0; i < TEST_TIME * K_FOLD; i++)
    {
        avg_precision += multi_test_accuracies[i].precision;
        avg_recall    += multi_test_accuracies[i].recall;
        avg_f1_score  += multi_test_accuracies[i].f1_score;
        avg_g_mean    += multi_test_accuracies[i].g_mean;
    }
    avg_precision       /= TEST_TIME * K_FOLD;
    avg_recall          /= TEST_TIME * K_FOLD;
    avg_f1_score        /= TEST_TIME * K_FOLD;
    avg_g_mean          /= TEST_TIME * K_FOLD;
    avg_elapsed_time_ms = elaped_time_ms / (TEST_TIME * K_FOLD);

    std::cout << avg_precision       << std::endl;
    std::cout << avg_recall          << std::endl;
    std::cout << avg_f1_score        << std::endl;
    std::cout << avg_g_mean          << std::endl;
    std::cout << avg_elapsed_time_ms << std::endl;
}