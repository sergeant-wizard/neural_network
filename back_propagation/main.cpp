#include<vector>
#include<functional>
#include<iostream>
#include<fstream>
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

int main(void) {
    std::ofstream ofs("residue.txt");
    srand(0);
    ActivationFunction rectifier(
        MatrixFunction([](double input) {
            return std::max<double>(input, 0);
        }),
        MatrixFunction([](double input) {
            if (input < 0)
                return 0;
            else
                return 1;
        }));
    ActivationFunction identity(
        MatrixFunction([](double input) {
            return input;
        }),
        MatrixFunction([](double) {
            return 1;
        }));

    const int firstNodeNum = 2;
    FirstLayer firstLayer(numBatch, firstNodeNum, rectifier);
    Layer midLayer(numBatch, 3, rectifier, &firstLayer);
    LastLayer lastLayer(numBatch, 2, identity, &midLayer);

    Matrix input(firstNodeNum, numBatch);
    input(0, 0) = +1;
    input(1, 0) = -1;
    input(0, 1) = -1;
    input(1, 1) = +1;
    input(0, 2) = +1;
    input(1, 2) = +1;

    Matrix target(firstNodeNum, numBatch);
    target(0, 0) = +1;
    target(1, 0) = -1;
    target(0, 1) = -1;
    target(1, 1) = +1;
    target(0, 2) = +1;
    target(1, 2) = +1;

    for (int i = 0; i < 64; i ++) {
        // forward propagation
        Matrix Y = lastLayer.forwardPropagation(
            midLayer.forwardPropagation(
                firstLayer.forwardPropagation(input)));
        lastLayer.setDelta(Y - target);

        // backward propagation
        Layer::backwardPropagation(midLayer, lastLayer);
        Layer::backwardPropagation(firstLayer, midLayer);

        // Gradient Descent
        Layer::gradientDescentForWeight(midLayer, lastLayer);
        Layer::gradientDescentForBias(lastLayer);
        Layer::gradientDescentForWeight(firstLayer, midLayer);
        Layer::gradientDescentForBias(midLayer);
        ofs << (Y - target).norm2() << std::endl;
    }

    // check learned result
    std::cout << "learned result:" << std::endl;
    lastLayer.forwardPropagation(
       midLayer.forwardPropagation(
           firstLayer.forwardPropagation(input))).print();
    return 0;
}
