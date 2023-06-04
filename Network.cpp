#include "Network.h"
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>

Network::Network(std::vector<int> sizes) :sizes(sizes), num_layers(sizes.size())
{

	srand(time(NULL));
	this->biases.resize(num_layers - 1);
	this->weights.resize(num_layers - 1);


	for (int val = 1; val < num_layers; ++val) {
		Eigen::VectorXd* currBias = new Eigen::VectorXd(this->sizes[val]);
		Eigen::MatrixXd* currWeight = new Eigen::MatrixXd(this->sizes[val], this->sizes[val - 1]);
		for (int y = 0; y < this->sizes[val]; ++y) {
			(*currBias)(y) = (double)rand() / (RAND_MAX / 2) - 1;
			for (int x = 0; x < this->sizes[val - 1]; ++x) {
				(*currWeight)(y, x) = (double)rand() / (RAND_MAX / 2) - 1;
			}
		}
		this->biases[val - 1] = currBias;
		this->weights[val - 1] = currWeight;
	}
}

Eigen::VectorXd Network::feedforward(Eigen::VectorXd inputs) {
	for (int i = 0; i < this->weights.size(); ++i) {
		inputs = network_analysis::sigmoid((*weights[i]) * inputs + (*biases[i]));
	}
	return inputs;
}

void Network::SGD(Eigen::MatrixXd* training_data, int epochs, int mini_batch_size, double eta, Eigen::MatrixXd* test_data) 
{
	int n_test = 0;
	if (test_data) {
		n_test = test_data->rows();
	}
	int n = training_data->rows();
	for (int j = 0; j < epochs; ++j) {
		std::random_device r;
		std::seed_seq rng_seed{r(), r(), r(), r(), r(), r(), r(), r()};
		std::mt19937 eng1(rng_seed);
		Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permX(training_data->rows());
		permX.setIdentity();
		std::shuffle(permX.indices().data(), permX.indices().data() + permX.indices().size(),eng1);
		(*training_data) = permX * (*training_data);
		std::vector<Eigen::MatrixXd> mini_batches(n/mini_batch_size);
		for (int i = 0; i < n / mini_batch_size; ++i) {
			Eigen::MatrixXd currBatch = training_data->middleRows(i * mini_batch_size, mini_batch_size);
			mini_batches[i] = currBatch;
		}
		int k = 1;
		printf("Epoch %d\n",j);
		printf("|                    |\n");
		printf(" ");

		for (Eigen::MatrixXd& mini_batch : mini_batches) {
			if (k % (mini_batches.size() / 20) == 0) {
				printf("=");
			}
			k++;
			update_mini_batch(mini_batch, eta);
		}
		printf("\n\n");
		if (test_data) {
			printf("Epoch %d: %d / %d\n\n", j, evaluate(test_data), n_test);
		}
		else{
			printf("Epoch %d complete", j);
		}
	}
	return;
}

void Network::update_mini_batch(Eigen::MatrixXd mini_batch, double eta)
{	
	int factorSize = this->weights.size();
	std::vector<Eigen::MatrixXd> nabla_b(factorSize);
	std::vector<Eigen::MatrixXd> nabla_w(factorSize);
	
	for (int i = 0; i < factorSize; ++i) {
		nabla_b[i] = Eigen::MatrixXd::Zero(this->biases[i]->rows(), this->biases[i]->cols());
		nabla_w[i] = Eigen::MatrixXd::Zero(this->weights[i]->rows(), this->weights[i]->cols());
	}
	for (int i = 0; i < mini_batch.rows(); ++i) {
		int expectedResult = mini_batch(i,0);
		Eigen::VectorXd x = Eigen::VectorXd::Zero(10);
		x(expectedResult) = 1;
		Eigen::VectorXd y = mini_batch.row(i).rightCols(784);
		y.adjointInPlace();
		std::pair<std::vector<Eigen::MatrixXd>,std::vector<Eigen::MatrixXd>> delta_nablas = backdrop(x, y);
		std::vector<Eigen::MatrixXd> delta_nabla_b = delta_nablas.first;
		std::vector<Eigen::MatrixXd> delta_nabla_w = delta_nablas.second;
		for (int set = 0; set < factorSize; ++set) {
			nabla_b[set] += delta_nabla_b[set];
			nabla_w[set] += delta_nabla_w[set];
		}
	}
	for (int set = 0; set < factorSize; ++set) {
		(*this->biases[set]) = (*this->biases[set]) - (eta / (double)mini_batch.size()) * nabla_b[set];
		(*this->weights[set]) = (*this->weights[set]) - (eta / (double)mini_batch.size()) * nabla_w[set];
	}
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> Network::backdrop(Eigen::VectorXd x, Eigen::VectorXd y)
{
	int factorSize = this->weights.size();
	std::vector<Eigen::MatrixXd> nabla_b(factorSize);
	std::vector<Eigen::MatrixXd> nabla_w(factorSize);

	
	for (int i = 0; i < factorSize; ++i) {
		nabla_b[i] = Eigen::MatrixXd::Zero(this->biases[i]->rows(), this->biases[i]->cols());
		nabla_w[i] = Eigen::MatrixXd::Zero(this->weights[i]->rows(), this->weights[i]->cols());
	}

	Eigen::VectorXd activation = y;
	std::vector<Eigen::VectorXd> activations = { activation };
	std::vector<Eigen::VectorXd> zs;
	
	for (int i = 0; i < factorSize; ++i) {
		Eigen::MatrixXd z = (*this->weights[i]) * activation + (*this->biases[i]);
		zs.push_back(z);
		activation = network_analysis::sigmoid(z);
		activations.push_back(activation);
	}
	Eigen::VectorXd lmul = cost_derivative(activations[activations.size() - 1], x);
	Eigen::VectorXd rmul = network_analysis::sigmoid_prime(zs[zs.size() - 1]); 
	Eigen::VectorXd delta = lmul.array() * rmul.array();
	nabla_b[nabla_b.size() - 1] = delta;
	for (int row = 0; row < delta.size(); ++row) {
		for (int col = 0; col < activations[activations.size() - 2].size(); ++col) {
			nabla_w[nabla_w.size() - 1](row,col) = delta(row) * activations[activations.size() - 2](col);
		}
	}
		
	for (int l = nabla_b.size() - 2; l >= 0; --l) {
		Eigen::MatrixXd z = zs[l];
		Eigen::VectorXd sp = network_analysis::sigmoid_prime(z);
		delta = ((*this->weights[l + 1]).adjoint() *delta).array() * sp.array();
		nabla_b[l] = delta;
		for (int row = 0; row < delta.size(); ++row) {
			for (int col = 0; col < activations[l].size(); ++col) {
				nabla_w[l](row,col) = delta(row) * activations[l](col);
			}
		}
	}
	return { nabla_b,nabla_w };
}

int Network::evaluate(Eigen::MatrixXd * test_data)
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
		int x = (*test_data)(i, 0);
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

Eigen::VectorXd Network::cost_derivative(Eigen::VectorXd output_activations, Eigen::VectorXd x)
{	
	return output_activations - x;
}



