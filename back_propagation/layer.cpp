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
    b(numNodes, 1),
    u(1, 1),
    z(1, 1),
    delta(1, 1)
{
    for (int i = 0; i < w.getRow() * w.getCol(); i++) {
        int dice = rand() % 10;
        w(i) = static_cast<double>(dice) / 10.0;
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

void Layer::gradientDescent(const Layer& prevLayer, Layer& nextLayer) {
    static const double epsilon = 0.1;
    Matrix DeltaW = Matrix::MultT2(nextLayer.delta, prevLayer.z);
    DeltaW *= epsilon / nextLayer.numBatch;
    nextLayer.w -= DeltaW;

    Matrix DeltaB(nextLayer.numBatch, 1);
    for (int nodeIndex = 0; nodeIndex < nextLayer.delta.getRow(); nodeIndex++) {
        double sum = 0;
        for (int batchIndex = 0; batchIndex < DeltaB.getRow(); batchIndex++) {
            sum += nextLayer.delta(nodeIndex, batchIndex);
        }
        DeltaB(nodeIndex, 0) = sum;
    }
    DeltaB *= epsilon / nextLayer.numBatch;
    nextLayer.b -= DeltaB;
}
void Layer::print() const {
    std::cout << "weight" << std::endl;
    w.print();
    std::cout << "bias" << std::endl;
    b.print();
    std::cout << "delta" << std::endl;
    delta.print();
}
