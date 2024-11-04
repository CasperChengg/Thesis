#include "../inc/file_operations.h"

#pragma region FUNCTION_DECLARATION
static void ReadDataset(std::vector<std::vector<float>> &dataset, std::string file_path);
static bool IsLabelExist(std::vector<uint32_t> &classes, uint32_t label);
static void GetNumClasses(Dataset &dataset);
static void Normalize(Dataset &dataset);
#pragma endregion // FUNCTION_DECLARATION

Dataset ReadTrainingAndTestingSet(std::string training_path, std::string testing_path)
{
    // Read training and testing set respectively
    Dataset dataset;
    ReadDataset(dataset.training_set, training_path);
    ReadDataset(dataset.testing_set,  testing_path);  

    GetNumClasses(dataset);
    Normalize(dataset);
    return dataset;
} 

static void ReadDataset(std::vector<std::vector<float>> &dataset, std::string file_path)
{
    // Open file
    std::ifstream file;
    file.open(file_path, std::ios::in);
    if (!file.is_open()){
        printf("./%s:%d: error: open file error\n", __FILE__, __LINE__);
        exit(1);
    }

    // Read file row by row
    std::string file_row;
    while (getline(file, file_row))
    {
        std::stringstream ss(file_row);
        std::string attribute;
        std::vector<float> data_row;
        // Read attributes from each row separated by "," into row vector
        while(getline(ss, attribute, ','))
        {
            data_row.push_back(std::stof(attribute.c_str()));
        }
        dataset.push_back(data_row);
    }
    file.close();
}
static bool IsLabelExist(std::vector<uint32_t> &classes, uint32_t label)
{
    for(uint32_t class_idx = 0; class_idx < classes.size(); class_idx++)
    {
        if(classes[class_idx] == label)
        {
            return true;
        }
    }
    return false;
}
static void GetNumClasses(Dataset &dataset)
{
    std::vector<uint32_t> classes;
    uint32_t data_label_idx = dataset.training_set[0].size() - 1;
    for(uint32_t data_idx = 0; data_idx < (dataset.training_set).size(); data_idx++)
    {
        uint32_t data_label = (dataset.training_set)[data_idx][data_label_idx];
        if(!IsLabelExist(classes, data_label))
        {
            classes.push_back(data_label);
        }
    }
    dataset.n_classes = classes.size();
}
static void Normalize(Dataset &dataset)
{
    // Normalized data without labels
    uint32_t n_dimensions = (dataset.training_set[0]).size() - 1;
    for(uint32_t dim_idx = 0; dim_idx < n_dimensions; dim_idx++)
    {
        // Find the maximum and minimum values across both the training and testing sets
        float max = 0, min = std::numeric_limits<float>::max();
        for(uint32_t training_idx = 0; training_idx < (dataset.training_set).size(); training_idx++)
        {
            if((dataset.training_set)[training_idx][dim_idx] > max)
            {
                max = (dataset.training_set)[training_idx][dim_idx];
            }
            if((dataset.training_set)[training_idx][dim_idx] < min)
            {
                min = (dataset.training_set)[training_idx][dim_idx];
            }
        }

        for(uint32_t testing_idx = 0; testing_idx < (dataset.testing_set).size(); testing_idx++)
        {
            if((dataset.testing_set)[testing_idx][dim_idx] > max)
            {
                max = (dataset.testing_set)[testing_idx][dim_idx];
            }
            if((dataset.testing_set)[testing_idx][dim_idx] < min)
            {
                min = (dataset.testing_set)[testing_idx][dim_idx];
            }
        }
        
        if(max == min)
        {
            for(uint32_t training_idx = 0; training_idx < (dataset.training_set).size(); training_idx++)
            {
                (dataset.training_set)[training_idx][dim_idx] = 0;
            }

            for(uint32_t testing_idx = 0; testing_idx < (dataset.testing_set).size(); testing_idx++)
            {
                (dataset.testing_set)[testing_idx][dim_idx] = 0;
            }  
        }
        else
        {
            for(uint32_t training_idx = 0; training_idx < (dataset.training_set).size(); training_idx++)
            {
                (dataset.training_set)[training_idx][dim_idx] = ((dataset.training_set)[training_idx][dim_idx] - min) / (max - min);;
            }

            for(uint32_t testing_idx = 0; testing_idx < (dataset.testing_set).size(); testing_idx++)
            {
                (dataset.testing_set)[testing_idx][dim_idx] = ((dataset.testing_set)[testing_idx][dim_idx] - min) / (max - min);
            }   
        }
    }
}
