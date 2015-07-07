#include "layer.h"
#include "matrix.h"
#include <iostream>
#include <stdlib.h>

Layer::Layer(
    const int numBatch,
    const int numNodes,
    const ActivationFunction& activationFunction,
    const Layer* prevLayer) :
    numBatch(numBatch),
    numNodes(numNodes),
    activationFunction(activationFunction),
    prevLayer(prevLayer),
    w(numNodes, prevLayer ? prevLayer->getNumNodes() : 1),
    prevDeltaW(numNodes, prevLayer ? prevLayer->getNumNodes() : 1),
    b(numNodes, 1),
    prevDeltaB(numNodes, 1),
    u(1, 1),
    z(1, 1),
    delta(1, 1)
{
    for (int i = 0; i < w.getRow() * w.getCol(); i++) {
        int dice = rand() % 20;
        w(i) = 0.4 + static_cast<double>(dice) / 100.0;
    }
}

int Layer::getNumNodes() const {
    return numNodes;
}

Matrix Layer::forwardPropagation(const Matrix& input) {
    // input : numNodes * numBatch
    // u : nextLayer->getNumNodes() * 1
    Matrix u = Matrix::Mult(w, input);
    for (int batchIndex = 0; batchIndex < numBatch; batchIndex++) {
    for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
        u(nodeIndex, batchIndex) += b(nodeIndex, 0);
    }}
    Matrix::swap(u, this->u);
    Matrix tmp = activationFunction.applyPrimaryFunction(this->u);
    Matrix::swap(tmp, z);
    return z;
}

void Layer::backwardPropagation(Layer& prevLayer, const Layer& nextLayer) {
    Matrix delta = Matrix::ComponentProduct(
        prevLayer.activationFunction.applyDerivativeFunction(prevLayer.u),
        Matrix::MultT1(nextLayer.w, nextLayer.delta));
    Matrix::swap(prevLayer.delta, delta);
}

void Layer::gradientDescentForWeight(const Layer& prevLayer, Layer& nextLayer) {
    static const double epsilon = 0.005; // learning rate
    static const double lambda = 0.1; // weight decay
    static const double mu = 0.5; // weight momentum

    // pure gradient descent
    Matrix DeltaW = Matrix::MultT2(nextLayer.delta, prevLayer.z);
    DeltaW *= -epsilon / nextLayer.numBatch;

    Matrix weightDecay = nextLayer.w;
    weightDecay *= -lambda * epsilon;

    Matrix momentum = nextLayer.prevDeltaW;
    momentum *= mu;

    nextLayer.prevDeltaW = DeltaW + weightDecay + momentum;

    nextLayer.w += nextLayer.prevDeltaW;
}

void Layer::gradientDescentForBias(Layer& nextLayer) {
    static const double epsilon = 0.01; // learning rate
    static const double mu = 0.5; // bias momentum
    Matrix DeltaB(nextLayer.numNodes, 1);
    for (int nodeIndex = 0; nodeIndex < nextLayer.delta.getRow(); nodeIndex++) {
        double sum = 0;
        for (int batchIndex = 0; batchIndex < DeltaB.getRow(); batchIndex++) {
            sum += nextLayer.delta(nodeIndex, batchIndex);
        }
        DeltaB(nodeIndex, 0) = sum;
    }
    DeltaB *= -epsilon / nextLayer.numBatch;

    Matrix momentum = nextLayer.prevDeltaB;
    momentum *= mu;

    nextLayer.prevDeltaB = DeltaB + momentum;
    nextLayer.b += nextLayer.prevDeltaB;
}

void Layer::print() const {
    std::cout << "weight" << std::endl;
    w.print();
    std::cout << "bias" << std::endl;
    b.print();
    std::cout << "delta" << std::endl;
    delta.print();
}
