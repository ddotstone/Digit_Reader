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
    void backdrop(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, std::vector<std::vector<Eigen::MatrixXd>>& delta_nabla_w,std::vector<Eigen::MatrixXd>& delta_nabla_b);
    int evaluate(const Eigen::MatrixXd* test_data);
    Eigen::MatrixXd cost_derivative(const Eigen::MatrixXd& output_activations, const Eigen::MatrixXd& x);
};

#endif
