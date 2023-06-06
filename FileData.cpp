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
    printf("|                    |\n");
    printf(" ");
    for (int row = 0; row < training_samples; ++row) {
        if (row % (training_samples / 20) == 0) {
            printf("=");
        }
        std::string line;
        std::getline(training_data_file, line);

        std::stringstream s;
        s << line;

        for (int i = 0;i < 785;i++) {
            std::string word;
            std::getline(s, word, ',');
            if (i == 0) {
                (*this->training_data)(row,i) = stoi(word) / 1.0;
            }
            else {
                (*this->training_data)(row,i) = stoi(word) / 255.0;

            }
        }
    }
    printf("\n");
    printf("Training Cases Complete\n\n");

    training_data_file.close();
    if (test_data == "") {
        this->test_data = NULL;
        printf("No Test Data Provided\n\n");
        return;
    }
    printf("Reading Test Data\n");
    printf("|                    |\n");
    printf(" ");

    assert(test_samples != -1);
	std::fstream test_data_file;
	test_data_file.open("./data/" + test_data, std::ios::in);
	
	assert(test_data_file.is_open());
    this->test_data = new Eigen::MatrixXd(test_samples, 785);
    for (int row = 0; row < test_samples; ++row) {

        if (row % (test_samples / 20) == 0) {
            printf("=");
        }


        std::string line;
        std::getline(test_data_file, line);

        std::stringstream s;
        s << line;

        for (int i = 0;i < 785;i++) {
            std::string word;
            std::getline(s, word, ',');
            if (i == 0) {
                (*this->test_data)(row,i) = stoi(word) / 1.0;
            }
            else {
                (*this->test_data)(row,i) = stoi(word) / 255.0;

            }
        }
    }

    printf("\n");
    printf("Test Cases Complete\n\n");
    test_data_file.close();
}

FileData::~FileData() {
    delete this->training_data;
    delete this->test_data;
}

Eigen::MatrixXd* FileData::getTraining_Data() const {
    return this->training_data;
}

Eigen::MatrixXd* FileData::getTest_Data() const {
    return this->test_data;
}



