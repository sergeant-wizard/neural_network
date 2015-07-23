#include<iostream>
#include<forward_list>
#include<random>
#include<cmath>
#include<map>

#include "matrix.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double gibbsSamplingForNode0(const Matrix& weight, const Matrix& bias, const Matrix& node) {
    const double expArg = bias(0, 0) + weight(0, 1) * node(1, 0);
    return sigmoid(expArg);
}

double gibbsSamplingForNode1(const Matrix& weight, const Matrix& bias, const Matrix& node) {
    const double expArg = bias(1, 0) + weight(0, 1) * node(0, 0);
    return sigmoid(expArg);
}

bool isProbable(double probability) {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(0, 1);
    return distribution(generator) < probability;
}

int main(void) {
    static const unsigned numSamples = 65536;
    static const unsigned numNodes = 2;
    // the weight matrix is redundant in this from,
    // since in Boltzmann machines they are symmetric.
    // we use only the right-upper part in this program.
    Matrix weight(numNodes, numNodes);
    weight(0, 1) = 0;
    Matrix bias(numNodes, 1);
    bias(0, 0) = 1;
    // latest node
    Matrix node(numNodes, 1);

    std::forward_list<Matrix> samples;
    for (unsigned int i = 0; i < numSamples; i++) {
        Matrix updatedNode = node;
        updatedNode(0, 0) = isProbable(gibbsSamplingForNode0(weight, bias, updatedNode)) ? 1.0 : 0.0;
        updatedNode(1, 0) = isProbable(gibbsSamplingForNode1(weight, bias, updatedNode)) ? 1.0 : 0.0;
        samples.push_front(updatedNode);
        std::swap(node, updatedNode);
    }

    std::map<unsigned, unsigned> nodeStateCount;
    for (const auto& node : samples) {
        const unsigned nodeKey = node(0) * 2 + node(1);
        nodeStateCount[nodeKey]++;
    }
    for (const auto& nodeStateCountPair : nodeStateCount) {
        std::cout
            << nodeStateCountPair.first
            << " "
            << static_cast<double>(nodeStateCountPair.second) / static_cast<double>(numSamples) * 100.0
            << std::endl;
    }

    return 0;
}
