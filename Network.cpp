#include "Network.h"
#include <cstdlib>
#include <ctime>

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
		for (int i = 0; i < n - 1; ++i) {
			int j = i + rand() % (n - i);
			Eigen::VectorXd temp = training_data->row(i);
			training_data->row(i) = training_data->row(j);
			training_data->row(j) = temp;
		}
		std::vector<Eigen::MatrixXd> mini_batches(n/mini_batch_size);
		for (int i = 0; i < n / mini_batch_size; ++i) {
			Eigen::MatrixXd currBatch = training_data->middleRows(i * mini_batch_size, mini_batch_size);
			mini_batches[i] = currBatch;
		}
		for (Eigen::MatrixXd& mini_batch : mini_batches) {
			update_mini_batch(mini_batch, eta);
		}
		if (test_data) {
			printf("Epoch %d: %d / %d", j, evaluate(test_data), n_test);
		}
		else{
			printf("Epoch %d complete", j);
		}
		return;
	}
}

void Network::update_mini_batch(Eigen::MatrixXd mini_batch, int eta)
{	
	int factorSize = this->weights.size();
	std::vector<Eigen::MatrixXd> nabla_b(factorSize);
	std::vector<Eigen::MatrixXd> nabla_w(factorSize);
	
	for (int i = 0; i < factorSize; ++i) {
		nabla_b[i] = Eigen::MatrixXd::Zero(this->biases[i]->rows(), this->biases[i]->cols());
		nabla_w[i] = Eigen::MatrixXd::Zero(this->weights[i]->rows(), this->weights[i]->cols());
	}
	for (int i = 0; i < mini_batch.size(); ++i) {
		int x = mini_batch(i,0);
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
		(*this->biases[set]) = (*this->biases[set]) - (eta / mini_batch.size()) * nabla_b[set];
		(*this->weights[set]) = (*this->weights[set]) - (eta / mini_batch.size()) * nabla_w[set];
	}
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> Network::backdrop(int x, Eigen::VectorXd y)
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
	nabla_w[nabla_w.size() - 1] = delta.adjoint() * (activations[activations.size() - 2]);
		
	for (int l = 1; l < this->num_layers; ++l) {
		Eigen::MatrixXd z = zs[zs.size() - 1 - l];
		Eigen::VectorXd sp = network_analysis::sigmoid_prime(z);
		delta = ((*this->weights[weights.size() - l]).adjoint() * delta) * sp;
		nabla_b[nabla_b.size() - l - 1] = delta;
		nabla_w[nabla_w.size() - l - 1] = delta.adjoint()*activations[activations.size() - l - 2];
	}
	return { nabla_b,nabla_w };
}

int Network::evaluate(Eigen::MatrixXd * test_data)
{
	std::vector<int> test_results;
	int correct_results = 0;
	for (int i = 0; i < test_data->size(); ++i) {
		int x = (*test_data)(i, 0);
		Eigen::VectorXd y = test_data->block(i, 1, 1, test_data->cols() - 1);
		Eigen::VectorXd currResult = feedforward(y);
		if (x == currResult.maxCoeff()) {
			correct_results += 1;
		}
	}
	return correct_results;
}

Eigen::VectorXd Network::cost_derivative(Eigen::VectorXd output_activations, int x)
{
	return output_activations - x * Eigen::VectorXd::Ones(output_activations.size());
}



