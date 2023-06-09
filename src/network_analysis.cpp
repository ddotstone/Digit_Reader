#include "network_analysis.h"
#include <cmath>

#include <iostream>

Eigen::MatrixXd network_analysis::sigmoid(Eigen::MatrixXd neurons) {
	for (int i = 0; i < neurons.rows(); ++i) {
		for (int j = 0; j < neurons.cols(); ++j) {
			neurons(i,j) = 1.0 / (1.0 + exp(-neurons(i,j)));
		}
	}
	return neurons;
}

Eigen::MatrixXd network_analysis::sigmoid_prime(Eigen::MatrixXd neurons) {
	Eigen::MatrixXd delta_neurons = sigmoid(neurons);
	for (int i = 0; i < neurons.rows(); ++i) {
		for (int j = 0; j < neurons.cols(); ++j) {
			neurons(i,j) = delta_neurons(i,j) * (1 - delta_neurons(i,j));
		}
	}
return neurons;
}

