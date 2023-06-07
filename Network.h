#ifndef NETWORK_H
#define NETWORK_H

#include <random>
#include <Eigen/Dense>
#include <iostream>

#include "network_analysis.h"

class Network {
private:
    //Network Sizes
    size_t num_layers;
    std::vector<int> sizes;

    //Weights between layers and layer biases
    std::vector<Eigen::VectorXd*> biases;
    std::vector<Eigen::MatrixXd*> weights;
public:
    //Constructors
    Network(std::vector<int> sizes);
    ~Network();

    //Filter Through Inputs
    Eigen::VectorXd feedforward(Eigen::VectorXd inputs);

    //Training model
    void SGD(Eigen::MatrixXd* training_data, int epochs, int mini_batch_size, double eta, const Eigen::MatrixXd* test_data);
    void update_mini_batch(const Eigen::MatrixXd* training_data, int start, int end, double eta);
    std::pair<std::vector<Eigen::MatrixXd>,std::vector<Eigen::MatrixXd>> backdrop(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
    int evaluate(const Eigen::MatrixXd* test_data);
    Eigen::VectorXd cost_derivative(const Eigen::VectorXd& output_activations, const Eigen::VectorXd& x);
};

#endif