#include "Network.h"
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <fstream>

Network::Network(std::vector<int> sizes) :sizes(sizes), num_layers(sizes.size())
{

	srand((int)time(NULL));
	this->biases.resize(num_layers - 1);
	this->weights.resize(num_layers - 1);


	for (int val = 1; val < num_layers; ++val) {
		Eigen::VectorXd* currBias = new Eigen::VectorXd(this->sizes[val]);
		Eigen::MatrixXd* currWeight = new Eigen::MatrixXd(this->sizes[val], this->sizes[val - 1]);
		for (int y = 0; y < this->sizes[val]; ++y) {
			(*currBias)(y) = ((rand() / (double)(RAND_MAX/2)) - 1);
			for (int x = 0;x < this->sizes[val - 1]; ++x) {
				(*currWeight)(y, x) = ((rand() / (double)(RAND_MAX / 2)) - 1);
			}
		}
		this->biases[val - 1] = currBias;
		this->weights[val - 1] = currWeight;
	}
}


Network::~Network() {
	for (int i = 0; i < this->num_layers - 1; ++i)
	{
		delete this->biases[i];
		delete this->weights[i];
	}
}

Eigen::VectorXd Network::feedforward(Eigen::VectorXd inputs) {
	for (int i = 0; i < this->weights.size(); ++i) {
		inputs = network_analysis::sigmoid((*weights[i]) * inputs + (*biases[i]));
	}
	return inputs;
}

void Network::SGD(Eigen::MatrixXd* training_data, int epochs, int mini_batch_size, double eta, const Eigen::MatrixXd* test_data) 
{
	int n_test = 0;
	if (test_data) {
		n_test = (int)test_data->rows();
	}
	int n = (int)training_data->rows();
	for (int j = 0; j < epochs; ++j) {
		std::random_device r;
		std::seed_seq rng_seed{r(), r(), r(), r(), r(), r(), r(), r()};
		std::mt19937 eng1(rng_seed);
		Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permX(training_data->rows());
		permX.setIdentity();
		std::shuffle(permX.indices().data(), permX.indices().data() + permX.indices().size(),eng1);
		(*training_data) = permX * (*training_data);
		
		printf("Epoch %d\n",j + 1);
		printf("|                    |\n");
		printf(" ");
		int k = 1;
		
		for (int i = 0; i < n / mini_batch_size; ++i) {
			int start = i * mini_batch_size;
			int end = start + mini_batch_size - 1;

			if ((k % (n/(mini_batch_size * 20))) == 0) {
				printf("=");
			}
			k++;
			update_mini_batch(training_data,start,end, eta);
		}
		printf("\n\n");
		if (test_data) {
			printf("Epoch %d: %d / %d\n\n", j + 1, evaluate(test_data), n_test);
		}
		else{
			printf("Epoch %d complete", j + 1);
		}
	}
	return;
}

void Network::update_mini_batch(const Eigen::MatrixXd* training_data,int start, int end, double eta)
{	
	int factorSize = (int)this->weights.size();
	std::vector<Eigen::MatrixXd> nabla_b(factorSize);
	std::vector<Eigen::MatrixXd> nabla_w(factorSize);
	
	for (int i = 0; i < factorSize; ++i) {
		nabla_b[i] = Eigen::MatrixXd::Zero(this->biases[i]->rows(), this->biases[i]->cols());
		nabla_w[i] = Eigen::MatrixXd::Zero(this->weights[i]->rows(), this->weights[i]->cols());
	}
	Eigen::MatrixXd x = Eigen::MatrixXd::Zero(10, end - start + 1);
	for (int i =start; i <= end; i++) {
		int expectedResult = (int)(*training_data)(i, 0);
		x(expectedResult, i - start) = 1;
	}
	Eigen::MatrixXd y = training_data->block(start,1,end - start + 1, 784);
	y.adjointInPlace();

	std::vector<Eigen::MatrixXd> delta_nabla_b(factorSize);
	std::vector<std::vector<Eigen::MatrixXd>> delta_nabla_w(factorSize, std::vector<Eigen::MatrixXd> (end - start + 1));

	backdrop(x, y,delta_nabla_w,delta_nabla_b);
	
	for (int set = 0; set < factorSize; ++set) {
		for (int i = 0; i < end - start + 1; ++i) {
			nabla_b[set] = nabla_b[set] + delta_nabla_b[set].col(i);
			nabla_w[set] = nabla_w[set] + delta_nabla_w[set][i];
		}
	}
	for (int set = 0; set < factorSize; ++set) {
		(*this->biases[set]) = (*this->biases[set]) - (eta / (double)(end - start + 1)) * nabla_b[set];
		(*this->weights[set]) = (*this->weights[set]) - (eta / (double)(end - start + 1)) * nabla_w[set];
	}
}

void Network::backdrop(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,std::vector<std::vector<Eigen::MatrixXd>>& nabla_w,std::vector<Eigen::MatrixXd>& nabla_b)
{
	int factorSize = (int)this->weights.size();
	int mini_batch_size = y.cols();

	Eigen::MatrixXd activation = y;
	std::vector<Eigen::MatrixXd> activations = { activation };
	std::vector<Eigen::MatrixXd> zs;
	for (int i = 0; i < factorSize; ++i) {
		Eigen::MatrixXd z = (*this->weights[i]) * activation;
		for (int j = 0; j < mini_batch_size; ++j) {
			z.col(j) += (*this->biases[i]);
		}
		zs.push_back(z);
		activation = network_analysis::sigmoid(z);
		activations.push_back(activation);
	}
	Eigen::ArrayXXd lmul = cost_derivative(activations[activations.size() - 1], x);
	Eigen::ArrayXXd rmul = network_analysis::sigmoid_prime(zs[zs.size() - 1]); 
	Eigen::MatrixXd delta = lmul * rmul;
	nabla_b[nabla_b.size() - 1] = delta;
	for (int i = 0; i < mini_batch_size; ++i) {
		nabla_w[nabla_w.size() - 1][i] = delta.col(i) * activations[activations.size() - 2].col(i).adjoint();
	}
	for (int l = (int)nabla_b.size() - 2; l >= 0; --l) {
		Eigen::MatrixXd z = zs[l];
		Eigen::ArrayXXd sp = network_analysis::sigmoid_prime(z);
		delta = ((*this->weights[l + 1]).adjoint() * delta).array() * sp;
		nabla_b[l] = delta;
		for (int i = 0; i < delta.cols(); ++i) {
			nabla_w[l][i] = delta.col(i) * activations[l].col(i).adjoint();
		}
	}
}

int Network::evaluate(const Eigen::MatrixXd * test_data)
{
	std::vector<int> test_results;
	int correct_results = 0;

	printf("Testing Results\n");
	printf("|                    |\n");
	printf(" ");

	for (int i = 0; i < test_data->rows(); ++i) {
		if (i % (test_data->rows() / 20) == 0) {
			printf("=");
		}
		int x = (int)(*test_data)(i, 0);
		Eigen::VectorXd y = test_data->row(i).rightCols(784);
		Eigen::VectorXd currResult = feedforward(y);
		Eigen::Index result;
		currResult.maxCoeff(&result);
		if (x == result) {
			correct_results += 1;
		}
	}
	printf("\n");
	printf("Testing Complete\n\n");
	return correct_results;
}

Eigen::MatrixXd Network::cost_derivative(const Eigen::MatrixXd& output_activations, const Eigen::MatrixXd& x)
{	
	return output_activations - x;
}

void Network::printResults() {
	std::ofstream resultFile("Results/results.csv");
	if (!resultFile.is_open()) {
		throw ("RESULTS.TXT_COULD_NOT_OPEN");
	}
	resultFile << "Weights,\n\n";
	for (int i = 0; i < this->weights.size(); ++i) {
		resultFile << "Layer " << i + 1 << " : " << i + 2 << "\n\n";
		for (int row = 0; row < this->weights[i]->rows();++row) {
			for (int col = 0; col < this->weights[i]->cols();++col) {
				resultFile << (*this->weights[i])(row, col) << ",";
			}
			resultFile << "\n";
		}
		resultFile << "\n";
	}
	resultFile << "Biases\n\n";
	for (int i = 0; i < this->biases.size(); ++i) {
		resultFile << "Layer " << i + 1 << " : " << i + 2 << "\n\n";
		for (int row = 0; row < this->biases[i]->rows();++row) {
			resultFile << (*this->biases[i])(row) << ",";
		}
		resultFile << "\n\n";
	}

	resultFile.close();
	return;
}

