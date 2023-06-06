#include "network_analysis.h"
#include <cmath>

#include <iostream>

Eigen::VectorXd network_analysis::sigmoid(Eigen::VectorXd neurons) {
	for (int i = 0; i < neurons.size(); ++i) {
		neurons(i) = 1.0 / (1.0 + exp(-neurons(i)));
	}
	return neurons;
}

Eigen::VectorXd network_analysis::sigmoid_prime(Eigen::VectorXd neurons) {
	Eigen::VectorXd delta_neurons = sigmoid(neurons);
	for (int i = 0; i < neurons.size(); ++i) {
		neurons(i) = delta_neurons(i) * (1 - delta_neurons(i));
	}
return neurons;
}

