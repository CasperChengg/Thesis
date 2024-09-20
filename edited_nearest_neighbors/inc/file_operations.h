#ifndef FILE_OPERATION_H
#define FILE_OPERATION_H

#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include "basic_structures.h"

Dataset ReadDataset(std::string dataset_path);
void GetDatasetInfo(Dataset *dataset, bool update);
void Normalize(Dataset *training_set, Dataset *testing_set);

#endif