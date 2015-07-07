#include<vector>
#include<functional>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<math.h>

#include "matrix.h"
#include "activation_function.h"
#include "layer.h"
#include "mnist_io.h"

int main(void) {
    std::ofstream ofs("residue.txt");
    srand(0);
    ActivationFunction rectifier(
        MatrixFunction([](double input) {
            return 1.0 / (1.0 + exp(input));
        }),
        MatrixFunction([](double input) {
            double exp_ = exp(input);
            return exp_ / (1.0 + exp_) / (1.0 + exp_);
        }));
    ActivationFunction identity(
        MatrixFunction([](double input) {
            return input;
        }),
        MatrixFunction([](double) {
            return 1;
        }));

    const int numBatches = 100;
    const int numSamples = 100;
    const int firstLayerNodeNum = MnistIO::numRows * MnistIO::numCols;
    const int midLayerNodeNum = 100;
    FirstLayer firstLayer(numBatches, firstLayerNodeNum, rectifier);
    Layer midLayer(numBatches, midLayerNodeNum, rectifier, &firstLayer);
    LastLayer lastLayer(numBatches, firstLayerNodeNum, identity, &midLayer);

    std::vector<Matrix> inputs = MnistIO::MnistToMatrixInBatches("../mnist_loader/t10k-images-idx3-ubyte", numBatches);

    Matrix average(firstLayerNodeNum, 1);
    for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
    for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
    for (int pixelIndex = 0; pixelIndex < firstLayerNodeNum; pixelIndex++) {
        average(pixelIndex) += inputs.at(batchIndex)(pixelIndex, sampleIndex) / numBatches / numSamples;
    }}}

    for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
    for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
    for (int pixelIndex = 0; pixelIndex < firstLayerNodeNum; pixelIndex++) {
        inputs.at(batchIndex)(pixelIndex, sampleIndex) -= average(pixelIndex);
    }}}

    for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
        const Matrix& input = inputs.at(batchIndex);
        std::cout << "batch index" << batchIndex << std::endl;
        // forward propagation
        Matrix Y = lastLayer.forwardPropagation(
            midLayer.forwardPropagation(
                firstLayer.forwardPropagation(input)));
        lastLayer.setDelta(Y - input);

        // backward propagation
        Layer::backwardPropagation(midLayer, lastLayer);
        Layer::backwardPropagation(firstLayer, midLayer);

        // Gradient Descent
        Layer::gradientDescentForWeight(midLayer, lastLayer);
        Layer::gradientDescentForBias(lastLayer);
        Layer::gradientDescentForWeight(firstLayer, midLayer);
        Layer::gradientDescentForBias(midLayer);
        std::cout << batchIndex << " " << (Y - input).norm2() << std::endl;
    }

    Matrix outputSample = lastLayer.forwardPropagation(
       midLayer.forwardPropagation(
           firstLayer.forwardPropagation(inputs.front())));
    MnistIO::MatrixToPNG(outputSample, 0);
    return 0;
}
