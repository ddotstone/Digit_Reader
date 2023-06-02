#include "FileData.h"
#include <fstream>
#include <iostream>
#include <cassert>

FileData::FileData(std::string training_data, int training_samples, std::string test_data, int test_samples) {
	std::fstream training_data_file;
	training_data_file.open("./data/" + training_data, std::ios::in);
	assert(training_data_file.is_open());
    this->training_data = new Eigen::MatrixXd(training_samples, 785);
    printf("Reading Training Data\n");
    for (unsigned int row = 0; row < training_samples; ++row) {
        if (row % (training_samples / 10) == 0) {
            printf("Reading Training Cases: %d%%\n", (row * 100) / training_samples);
        }
        std::string line;
        std::getline(training_data_file, line);

        std::stringstream s;
        s << line;

        for (int i = 0;i < 785;i++) {
            std::string word;
            std::getline(s, word, ',');
            (*this->training_data)(row,i) = (i == 0)?stoi(word):stoi(word) / 255.0;
        }
    }
    printf("Training Cases Complete\n");
    training_data_file.close();
    if (test_data == "") {
        this->test_data = NULL;
        printf("No Test Data Provided\n");
        return;
    }
    printf("Reading Test Data\n");
    assert(test_samples != -1);
	std::fstream test_data_file;
	test_data_file.open("./data/" + test_data, std::ios::in);
	
	assert(test_data_file.is_open());
    this->test_data = new Eigen::MatrixXd(test_samples, 785);
    for (int row = 0; row < test_samples; ++row) {

        if (row % (test_samples / 10) == 0) {
            printf("Reading Test Cases: %d%%\n", (row * 100) / test_samples);
        }


        std::string line;
        std::getline(test_data_file, line);

        std::stringstream s;
        s << line;

        for (int i = 0;i < 785;i++) {
            std::string word;
            std::getline(s, word, ',');
            (*this->training_data)(row,i) = (i == 0)?stoi(word):stoi(word) / 255.0;
        }
    }

    printf("Training Cases Complete\n");

    test_data_file.close();

}

Eigen::MatrixXd* FileData::getTraining_Data() const {
    return this->training_data;
}

Eigen::MatrixXd* FileData::getTest_Data() const {
    return this->test_data;
}



