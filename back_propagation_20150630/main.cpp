#include<vector>
#include<functional>
#include<iostream>
#include<algorithm>

#include "matrix.h"
#include "activation_function.h"
#include "layer.h"

/*
 * w : weight
 * b : bias
 * u : input
 * y : output
 * d : target output
 */

const int numBatch = 3;
const int firstNodeNum = 2;

void applyIteration(FirstLayer& firstLayer, LastLayer& secondLayer) {
    // forward propagation
    Matrix input(firstNodeNum, numBatch);
    input(0, 0) = +1;
    input(1, 0) = -1;
    input(0, 1) = -1;
    input(1, 1) = +1;
    input(0, 2) = +1;
    input(1, 2) = +1;

    Matrix Y = secondLayer.forwardPropagation(firstLayer.forwardPropagation(input));

    // backward propagation

    /*
    // target output (identity mapping)
    Matrix target(firstNodeNum, numBatch);
    target(0, 0) = +1;
    target(1, 0) = +0;
    target(0, 1) = -1;
    target(1, 1) = +0;
    target(0, 2) = +1;
    target(1, 2) = +0;
    */
    /*
    // target output (AND circuit)
    Matrix target(firstNodeNum, numBatch);
    target(0, 0) =  0;
    target(1, 0) =  0;
    target(0, 1) =  0;
    target(1, 1) =  0;
    target(0, 2) = +1;
    target(1, 2) =  0;
    */
    // target output (XOR circuit)
    Matrix target(firstNodeNum, numBatch);
    target(0, 0) =  1;
    target(1, 0) =  0;
    target(0, 1) =  1;
    target(1, 1) =  0;
    target(0, 2) =  0;
    target(1, 2) =  0;

    secondLayer.setDelta(Y - target);
    Layer::backwardPropagation(firstLayer, secondLayer);

    // Gradient Descent
    Layer::gradientDescent(firstLayer, secondLayer);
    secondLayer.print();
}

int main(void) {
    ActivationFunction activationFunction(
        MatrixFunction([](double input) {
            return std::max<double>(input, 0);
        }),
        MatrixFunction([](double input) {
            if (input < 0)
                return 0;
            else
                return 1;
        }));

    FirstLayer firstLayer(numBatch, firstNodeNum, activationFunction);
    LastLayer secondLayer(numBatch, firstNodeNum, activationFunction, &firstLayer);

    for (int i = 0; i < 64; i ++) {
        applyIteration(firstLayer, secondLayer);
    }
    return 0;
}
