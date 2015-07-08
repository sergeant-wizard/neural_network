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

    const int numBatches = 100;
    const int numSamples = 100;
    const int numIterations = 100;
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

    for (int iterationIndex = 0; iterationIndex < numIterations; iterationIndex++) {
        for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
            const Matrix& input = inputs.at(batchIndex);
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
            std::cout << iterationIndex << " " << batchIndex << " " << (Y - input).norm2() << std::endl;
        }
        Matrix outputSample = lastLayer.forwardPropagation(
           midLayer.forwardPropagation(
               firstLayer.forwardPropagation(inputs.front())));
        for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
        for (int pixelIndex = 0; pixelIndex < firstLayerNodeNum; pixelIndex++) {
            outputSample(pixelIndex, sampleIndex) += average(pixelIndex);
        }}
        MnistIO::MatrixToPNG(outputSample, 0);
    }

    return 0;
}
