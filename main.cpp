#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>

#include "Network.h"
#include "FileData.h"

const int PIXELS = 784;
const int HIDDEN_LAYERS = 30;
const int OUTPUTS = 10;
const int EPOCHS = 30;
const int MINI_BATCH_SIZE = 10;
const double LEARNING_RATE = 3.0;

int main(int argc, char* argv[]) {
	assert(argc == 2);
	std::string input_file = argv[1];

	std::fstream inputs;

	inputs.open("./inputs/" + input_file, std::ios::in);

	std::string line;

	std::string training_data_file;
	int training_samples;
	std::string test_data_files = "";
	int test_samples = -1;
		
	getline(inputs, line);
	training_data_file = line;
	
	getline(inputs, line);
	training_samples = stoi(line);

	if (getline(inputs, line)) {
		test_data_files = line;
		assert(getline(inputs, line));
		test_samples = stoi(line);
	}

	FileData data = FileData(training_data_file, training_samples, test_data_files, test_samples);

	Network network({PIXELS,HIDDEN_LAYERS,OUTPUTS});

	network.SGD(data.getTraining_Data(), EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, data.getTest_Data());

}
