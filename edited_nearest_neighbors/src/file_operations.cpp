#include "file_operations.h"

bool LabelExist(std::vector<int> &classes, size_t label)
{
    for(size_t i = 0; i < classes.size(); i++)
    {
        if(classes[i] == label)
        {
            return true;
        }
    }
    return false;
}

void GetDatasetInfo(Dataset *dataset, bool update)
{
    if((dataset->training_set)[0].size() == (dataset->testing_set)[0].size())
    {
        dataset->dimension   = (dataset->training_set)[0].size() - 1;
    }
    else
    {
        printf("./%s:%d: error: the dimension of the training and testing set do not match\n", __FILE__, __LINE__);
        exit(1);
    }
    
    // assume labels are stored in the last column of the dataset
    dataset->label_index = (dataset->training_set)[0].size() - 1;
#pragma region COUNT_NUM_SAMPLES_PER_CLASS 
    // calculate training set class counts
    if(!update)
    {
        std::vector<int> classes;
        for(size_t i = 0; i < (dataset->training_set).size(); i++)
        {
            size_t label = (dataset->training_set)[i][dataset->label_index];
            if(!LabelExist(classes, label))
            {
                classes.push_back(label);
            }
        }
        dataset->num_classes  = classes.size();
        dataset->training_set_class_counts = (size_t*)calloc(dataset->num_classes + 1, sizeof(size_t));
    }
    else
    {
        memset(dataset->training_set_class_counts, 0, dataset->num_classes + 1 * sizeof(size_t));
    }

    for(size_t i = 0; i < (dataset->training_set).size(); i++)
    {
        size_t label = (dataset->training_set)[i][dataset->label_index];
        (dataset->training_set_class_counts)[label]++;
    }

    // calculate testing set class counts
    dataset->testing_set_class_counts = (size_t*)calloc(dataset->num_classes + 1, sizeof(size_t));
    for(size_t i = 0; i < (dataset->testing_set).size(); i++)
    {
        size_t label = (dataset->testing_set)[i][dataset->label_index];
        (dataset->testing_set_class_counts)[label]++;
    }
#pragma endregion

}

void Normalize(Dataset *dataset)
{
    for(size_t dim = 0; dim < dataset->dimension; dim++)
    {
        float max = (dataset->training_set)[0][dim];
        float min = (dataset->training_set)[0][dim];
        for(size_t i = 0; i < (dataset->training_set).size(); i++)
        {
            if((dataset->training_set)[i][dim] > max)
            {
                max = (dataset->training_set)[i][dim];
            }
            if((dataset->training_set)[i][dim] < min)
            {
                min = (dataset->training_set)[i][dim];
            }
        }

        for(size_t i = 0; i < (dataset->testing_set).size(); i++)
        {
            if((dataset->testing_set)[i][dim] > max)
            {
                max = (dataset->testing_set)[i][dim];
            }
            if((dataset->testing_set)[i][dim] < min)
            {
                min = (dataset->testing_set)[i][dim];
            }
        }

        if(max == min)
        {
            for(size_t i = 0; i < (dataset->training_set).size(); i++)
            {
                (dataset->training_set)[i][dim] = 0;
            } 
            for(size_t i = 0; i < (dataset->testing_set).size(); i++)
            {
                (dataset->testing_set)[i][dim] = 0;
            }  
        }
        else
        {
            for(size_t i = 0; i < (dataset->training_set).size(); i++)
            {
                (dataset->training_set)[i][dim] = ((dataset->training_set)[i][dim] - min) / (max - min);
            } 
            for(size_t i = 0; i < (dataset->testing_set).size(); i++)
            {
                (dataset->testing_set)[i][dim] = ((dataset->testing_set)[i][dim] - min) / (max - min);
            }   
        }
    }
}

Dataset ReadDataset(std::string dataset_path)
{
    Dataset dataset;
    std::stringstream ss;
    std::ifstream file;
    std::string file_row, attribute;
    std::vector<float> data_row;

#pragma region READ_TRAINING_SET
    std::string training_path = dataset_path + "tra.dat";
    file.open(training_path, std::ios::in);
    if (!file.is_open()){
        printf("./%s:%d: error: open file error\n", __FILE__, __LINE__);
        exit(1);
    }

    while (getline(file, file_row))
    {
        ss.str(file_row);
        while(getline(ss, attribute, ','))
        {
            data_row.push_back(std::stof(attribute.c_str()));
        }
        (dataset.training_set).push_back(data_row);
        
        ss.clear();
        data_row.clear();
    }
    file.close();
#pragma endregion

#pragma region READ_TESTING_SET
    std::string testing_path = dataset_path + "tst.dat";
    file.open(testing_path, std::ios::in);
    if (!file.is_open()){
        printf("./%s:%d: \033[31merror\033[0m: open file error\n", __FILE__, __LINE__);
        exit(1);
    }

    while (getline(file, file_row))
    {
        ss.str(file_row);
        while(getline(ss, attribute, ','))
        {
            data_row.push_back(std::stof(attribute.c_str()));
        }
        dataset.testing_set.push_back(data_row);

        ss.clear();
        data_row.clear();
    }
    file.close();
#pragma endregion

    GetDatasetInfo(&dataset, false);
    Normalize(&dataset);
    return dataset;
} 