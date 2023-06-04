#ifndef NETWORK_H
#define NETWORK_H

#include <random>
#include <Eigen/Dense>
#include <iostream>

#include "network_analysis.h"

class Network {
private:
    //Network Sizes
    int num_layers;
    std::vector<int> sizes;

    //Weights between layers and layer biases
    std::vector<Eigen::VectorXd*> biases;
    std::vector<Eigen::MatrixXd*> weights;
public:
    //Constructors
    Network(std::vector<int> sizes);

    //Filter Through Inputs
    Eigen::VectorXd feedforward(Eigen::VectorXd inputs);

    //Training model
    void SGD(Eigen::MatrixXd* training_data, int epochs, int mini_batch_size, double eta, Eigen::MatrixXd* test_data);
    void update_mini_batch(Eigen::MatrixXd mini_batch, double eta);
    std::pair<std::vector<Eigen::MatrixXd>,std::vector<Eigen::MatrixXd>> backdrop(Eigen::VectorXd x, Eigen::VectorXd y);
    int evaluate(Eigen::MatrixXd* test_data);
    Eigen::VectorXd cost_derivative(Eigen::VectorXd output_activations, Eigen::VectorXd x);
};

#endif