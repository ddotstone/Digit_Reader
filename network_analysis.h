#pragma once

#include <string> 
#include<Eigen/Dense>

namespace network_analysis {
	Eigen::MatrixXd sigmoid(Eigen::MatrixXd neurons);
	Eigen::MatrixXd sigmoid_prime(Eigen::MatrixXd neurons);
}

