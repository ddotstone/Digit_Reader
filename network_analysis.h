#pragma once

#include <string> 
#include<Eigen/Dense>

namespace network_analysis {
	Eigen::VectorXd sigmoid(Eigen::VectorXd neurons);
	Eigen::VectorXd sigmoid_prime(Eigen::VectorXd neurons);
}

