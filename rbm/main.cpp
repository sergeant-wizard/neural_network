#include<vector>
#include<random>
#include<functional>
#include<iostream>
#include<fstream>
#include<algorithm>
#include<cmath>
#include<forward_list>

#include "matrix.h"

class RBM {
public:
    struct UpdatedParams {
        Matrix weight;
        Matrix visibleBias;
        Matrix hiddenBias;
    };
    RBM(unsigned numVisibleNodes, unsigned numHiddenNodes);
    UpdatedParams getModelUpdateParams() const;
    UpdatedParams getDataUpdateParams(const Matrix& input) const;

private:
    static const unsigned numGibbsSamples = 16;
    const unsigned numVisibleNodes;
    const unsigned numHiddenNodes;

    static double sigmoid(double);
    static double decideActivation(double probability);

    Matrix weight;
    Matrix visibleNode;
    Matrix hiddenNode;
    Matrix visibleBias;
    Matrix hiddenBias;
};

RBM::RBM(unsigned numVisibleNodes, unsigned numHiddenNodes) :
    numVisibleNodes(numVisibleNodes),
    numHiddenNodes(numHiddenNodes),
    weight(numVisibleNodes, numHiddenNodes),
    visibleNode(numVisibleNodes, 1),
    hiddenNode(numHiddenNodes, 1),
    visibleBias(numVisibleNodes, 1),
    hiddenBias(numHiddenNodes, 1)
{
}

RBM::UpdatedParams RBM::getModelUpdateParams() const {
    UpdatedParams ret {
        Matrix(numVisibleNodes, numHiddenNodes),
        Matrix(numVisibleNodes,1),
        Matrix(numHiddenNodes,1)
    };
    Matrix visibleNode = this->visibleNode;
    Matrix hiddenNode = this->hiddenNode;
    // Gibbs sampling
    for (unsigned iterationIndex = 0; iterationIndex < numGibbsSamples; iterationIndex++) {
        // update visible layer
        for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numHiddenNodes; visibleNodeIndex++) {
            double sigmoidArg = visibleBias(visibleNodeIndex);
            for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numVisibleNodes; hiddenNodeIndex++) {
                sigmoidArg += weight(visibleNodeIndex, hiddenNodeIndex) * hiddenNode(hiddenNodeIndex);
            }
            visibleNode(visibleNodeIndex) = decideActivation(sigmoid(sigmoidArg));
        }
        // update hidden layer
        for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numHiddenNodes; hiddenNodeIndex++) {
            double sigmoidArg = hiddenBias(hiddenNodeIndex);
            for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
                sigmoidArg += weight(visibleNodeIndex, hiddenNodeIndex) * visibleNode(visibleNodeIndex);
            }
            hiddenNode(hiddenNodeIndex) = decideActivation(sigmoid(sigmoidArg));
        }
        ret.visibleBias += visibleNode;
        ret.hiddenBias += hiddenNode;
        ret.weight += Matrix::MultT2(visibleNode, hiddenNode);
    }
    const double coef = 1.0 / static_cast<double>(numGibbsSamples);
    ret.visibleBias *= coef;
    ret.hiddenBias *= coef;
    ret.weight *= coef;
    return ret;
}

RBM::UpdatedParams RBM::getDataUpdateParams(const Matrix& input) const {
    UpdatedParams ret {
        Matrix(numVisibleNodes, numHiddenNodes),
        Matrix(numVisibleNodes,1),
        Matrix(numHiddenNodes,1)
    };
    for (int dataIndex = 0; dataIndex < input.getCol(); dataIndex++) {
        double sigmoidArg = hiddenBias(dataIndex);
    }
    return ret;
}

double RBM::sigmoid(const double input) {
    return 1.0 / (1.0 + exp(-input));
}

double RBM::decideActivation(double probability) {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution(0, 1);
    return distribution(generator) < probability ? 1.0 : 0.0;
}

int main(void) {
    static const unsigned numVisibleNodes = 2;
    static const unsigned numHiddenNodes = 2;
    static const unsigned numInputData = 2;
    Matrix input(numVisibleNodes, numInputData);

    {
        input(0, 0) = 0;
        input(0, 1) = 0;
        input(1, 0) = 0;
        input(1, 1) = 0;
    }

    RBM rbm(numVisibleNodes, numHiddenNodes);
    rbm.getModelUpdateParams();

    return 0;
}
