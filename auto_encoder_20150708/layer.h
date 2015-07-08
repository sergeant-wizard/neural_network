#pragma once
#include "activation_function.h"
class Matrix;

class Layer {
public:
    Layer(
        const int numBatch,
        const int numNodes,
        const ActivationFunction& activationFunction,
        const Layer* prevLayer);

    int getNumNodes() const;
    virtual Matrix forwardPropagation(const Matrix& input);
    static void backwardPropagation(Layer& prevLayer, const Layer& nextLayer);
    static void gradientDescentForWeight(const Layer& prevLayer, Layer& nextLayer);
    static void gradientDescentForBias(Layer& nextLayer);
    void print() const;

protected:
    const int numBatch;
    const int numNodes;
    const ActivationFunction activationFunction;
    const Layer* prevLayer;
    // weight
    Matrix w;
    Matrix prevDeltaW;
    // bias
    Matrix b;
    Matrix prevDeltaB;

    Matrix u;
    Matrix z;
    Matrix delta;
};

class FirstLayer : public Layer {
public:
    FirstLayer(int numBatch, int numNodes, const ActivationFunction& activationFunction) :
        Layer(
            numBatch,
            numNodes,
            activationFunction,
            nullptr)
    {
    }
    Matrix forwardPropagation(const Matrix& input) override {
        // identity mapping
        return u = z = input;
    }
};

class LastLayer : public Layer {
public:
    using Layer::Layer;
    void setDelta(Matrix&& delta) {
        Matrix::swap(this->delta, delta);
    }
};
