#pragma once

#include <Eigen/Dense>
#include <string>

class FileData
{
private:
	Eigen::MatrixXd* training_data;
	Eigen::MatrixXd* test_data;
public:
	FileData(std::string training_data, int samples, std::string test_data, int test_samples);
	~FileData();
	
	//Getters
	Eigen::MatrixXd* getTraining_Data() const;
	Eigen::MatrixXd* getTest_Data() const;
};

