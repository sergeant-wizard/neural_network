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
        void normalize(int numGibbsSamples) {
            const double coef = 1.0 / static_cast<double>(numGibbsSamples);
            visibleBias *= coef;
            hiddenBias *= coef;
            weight *= coef;
        }
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

    double getVisiblePosteriorProbability(unsigned visibleNodeIndex, const Matrix& hiddenNode) const;
    double getHiddenPosteriorProbability(unsigned hiddenNodeIndex, const Matrix& visibleNode) const;

    Matrix weight;
    // Matrix visibleNode;
    // Matrix hiddenNode;
    Matrix visibleBias;
    Matrix hiddenBias;
};

RBM::RBM(unsigned numVisibleNodes, unsigned numHiddenNodes) :
    numVisibleNodes(numVisibleNodes),
    numHiddenNodes(numHiddenNodes),
    weight(numVisibleNodes, numHiddenNodes),
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
    Matrix visibleNode(numVisibleNodes, 1);
    Matrix hiddenNode(numHiddenNodes, 1);
    // Gibbs sampling
    for (unsigned iterationIndex = 0; iterationIndex < numGibbsSamples; iterationIndex++) {
        // update visible layer
        for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numHiddenNodes; visibleNodeIndex++) {
            visibleNode(visibleNodeIndex) = decideActivation(getVisiblePosteriorProbability(visibleNodeIndex, hiddenNode));
        }
        // update hidden layer
        for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numHiddenNodes; hiddenNodeIndex++) {
            hiddenNode(hiddenNodeIndex) = decideActivation(getHiddenPosteriorProbability(hiddenNodeIndex, visibleNode));
        }
        ret.visibleBias += visibleNode;
        ret.hiddenBias += hiddenNode;
        ret.weight += Matrix::MultT2(visibleNode, hiddenNode);
    }
    ret.normalize(numGibbsSamples);
    return ret;
}

RBM::UpdatedParams RBM::getDataUpdateParams(const Matrix& input) const {
    UpdatedParams ret {
        Matrix(numVisibleNodes, numHiddenNodes),
        Matrix(numVisibleNodes,1),
        Matrix(numHiddenNodes,1)
    };
    Matrix visibleNode(numVisibleNodes, 1);
    Matrix hiddenNode(numHiddenNodes, 1);
    const unsigned numInputData = input.getCol();
    for (unsigned dataIndex = 0; dataIndex < numInputData; dataIndex++) {
        for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
            visibleNode(visibleNodeIndex, 0) = input(visibleNodeIndex, dataIndex);
        }
        // weight, hidden bias
        for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numHiddenNodes; hiddenNodeIndex++) {
            const double hiddenPosteriorPriority = getHiddenPosteriorProbability(hiddenNodeIndex, visibleNode);
            ret.hiddenBias(hiddenNodeIndex, 0) += hiddenPosteriorPriority;
            for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
                ret.weight(visibleNodeIndex, hiddenNodeIndex) += hiddenPosteriorPriority * visibleNode(visibleNodeIndex);
            }
        }
        // visible bias
        ret.visibleBias += visibleNode;
    }
    ret.normalize(numInputData);
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

double RBM::getVisiblePosteriorProbability(unsigned visibleNodeIndex, const Matrix& hiddenNode) const {
    double sigmoidArg = visibleBias(visibleNodeIndex);
    for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numVisibleNodes; hiddenNodeIndex++) {
        sigmoidArg += weight(visibleNodeIndex, hiddenNodeIndex) * hiddenNode(hiddenNodeIndex);
    }
    return sigmoid(sigmoidArg);
}

double RBM::getHiddenPosteriorProbability(unsigned hiddenNodeIndex, const Matrix& visibleNode) const {
    double sigmoidArg = hiddenBias(hiddenNodeIndex);
    for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
        sigmoidArg += weight(visibleNodeIndex, hiddenNodeIndex) * visibleNode(visibleNodeIndex);
    }
    return sigmoid(sigmoidArg);
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
