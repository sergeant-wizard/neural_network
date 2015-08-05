#include<vector>
#include<random>
#include<iostream>
#include<cmath>
#include<iomanip>

#include "matrix.h"

class RBM {
public:
    struct UpdatedParams {
        Matrix weight;
        Matrix visibleBias;
        Matrix hiddenBias;
        void normalize(int num) {
            const double coef = 1.0 / static_cast<double>(num);
            visibleBias *= coef;
            hiddenBias *= coef;
            weight *= coef;
        }
    };
    RBM(unsigned numVisibleNodes, unsigned numHiddenNodes);
    void update(const UpdatedParams& modelUpdateParams, const UpdatedParams& dataUpdateParams);
    void updateByCD(const UpdatedParams& updateParams);
    UpdatedParams getModelUpdateParamsByGibbsSampling() const;
    UpdatedParams getUpdateParamsByCD(const Matrix& input) const;
    UpdatedParams getDataUpdateParams(const Matrix& input) const;

    // these methods are only for debugging and should not be used for the actual numerical procedure
    double getEnergy(const Matrix& visibleNode, const Matrix& hiddenNode) const;
    static std::vector<Matrix> getPowerSet(unsigned numNodes);

private:
    static const unsigned numGibbsSamples = 1024;
    static const unsigned CDIterations = 10;
    const unsigned numVisibleNodes;
    const unsigned numHiddenNodes;

    static double sigmoid(double);
    static double decideActivation(double probability);

    double getVisiblePosteriorProbability(unsigned visibleNodeIndex, const Matrix& hiddenNode) const;
    double getHiddenPosteriorProbability(unsigned hiddenNodeIndex, const Matrix& visibleNode) const;
    Matrix getHiddenPosteriorProbability(const Matrix& visibleNode) const;

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
    weight.randomize(-0.1, 0.1);
}

void RBM::update(const UpdatedParams& modelUpdateParams, const UpdatedParams& dataUpdateParams)
{
    static const double learningRate = 0.1;
    weight += ((dataUpdateParams.weight - modelUpdateParams.weight) *= learningRate);
    visibleBias += ((dataUpdateParams.visibleBias - modelUpdateParams.visibleBias) *= learningRate);
    hiddenBias += ((dataUpdateParams.hiddenBias - modelUpdateParams.hiddenBias) *= learningRate);
}

void RBM::updateByCD(const UpdatedParams& updatedParams)
{
    static const double learningRate = 0.01;
    Matrix weightUpdate = updatedParams.weight;
    Matrix visibleBiasUpdade = updatedParams.visibleBias;
    Matrix hiddenBiasUpdate = updatedParams.hiddenBias;

    weightUpdate *= learningRate;
    visibleBias *= learningRate;
    hiddenBiasUpdate *= learningRate;

    weight += weightUpdate;
    visibleBias += visibleBiasUpdade;
    hiddenBias += hiddenBiasUpdate;
}

RBM::UpdatedParams RBM::getModelUpdateParamsByGibbsSampling() const {
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
        for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
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

RBM::UpdatedParams RBM::getUpdateParamsByCD(const Matrix& input) const {
    UpdatedParams ret {
        Matrix(numVisibleNodes, numHiddenNodes),
        Matrix(numVisibleNodes,1),
        Matrix(numHiddenNodes,1)
    };
    const unsigned numInputData = input.getCol();

    for (unsigned dataIndex = 0; dataIndex < numInputData; dataIndex++) {
        Matrix visibleNode0(numVisibleNodes, 1);
        Matrix hiddenNode(numHiddenNodes, 1);

        for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
            visibleNode0(visibleNodeIndex) = input(visibleNodeIndex, dataIndex);
        }
        Matrix hiddenPosteriorPriority0 = getHiddenPosteriorProbability(visibleNode0);
        Matrix visibleNode = visibleNode0;

        for (unsigned iterationIndex = 0; iterationIndex < CDIterations; iterationIndex++) {
            // update visible layer
            for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
                visibleNode(visibleNodeIndex) = decideActivation(getVisiblePosteriorProbability(visibleNodeIndex, hiddenNode));
            }
            // update hidden layer
            for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numHiddenNodes; hiddenNodeIndex++) {
                hiddenNode(hiddenNodeIndex) = decideActivation(getHiddenPosteriorProbability(hiddenNodeIndex, visibleNode));
            }
        }

        const Matrix& visibleNodeT = visibleNode;
        Matrix hiddenPosteriorPriorityT = getHiddenPosteriorProbability(visibleNode);
        ret.weight += (Matrix::MultT2(visibleNode0, hiddenPosteriorPriority0) - Matrix::MultT2(visibleNodeT, hiddenPosteriorPriorityT));
        ret.visibleBias += visibleNode0 - visibleNodeT;
        ret.hiddenBias += hiddenPosteriorPriority0 - hiddenPosteriorPriorityT;
    }
    ret.normalize(numInputData);

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

double RBM::getEnergy(const Matrix& visibleNode, const Matrix& hiddenNode) const {
    double sum = 0;
    for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
        sum += visibleBias(visibleNodeIndex, 0) * visibleNode(visibleNodeIndex);
    }
    for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numHiddenNodes; hiddenNodeIndex++) {
        sum += hiddenBias(hiddenNodeIndex, 0) * hiddenNode(hiddenNodeIndex);
    }
    for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
        for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numHiddenNodes; hiddenNodeIndex++) {
            sum += weight(visibleNodeIndex, hiddenNodeIndex) * visibleNode(visibleNodeIndex) * hiddenNode(hiddenNodeIndex);
        }
    }
    return exp(sum);
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
    for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numHiddenNodes; hiddenNodeIndex++) {
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

Matrix RBM::getHiddenPosteriorProbability(const Matrix& visibleNode) const {
    Matrix ret(numHiddenNodes, 1);
    for (unsigned hiddenNodeIndex = 0; hiddenNodeIndex < numHiddenNodes; hiddenNodeIndex++) {
        double sigmoidArg = hiddenBias(hiddenNodeIndex);
        for (unsigned visibleNodeIndex = 0; visibleNodeIndex < numVisibleNodes; visibleNodeIndex++) {
            sigmoidArg += weight(visibleNodeIndex, hiddenNodeIndex) * visibleNode(visibleNodeIndex);
        }
        ret(hiddenNodeIndex, 0) = sigmoid(sigmoidArg);
    }
    return ret;
}

std::vector<Matrix> RBM::getPowerSet(unsigned numNodes) {
    std::vector<Matrix> ret;
    for (int i = 0; i < pow(2, numNodes); i++) {
        Matrix newNode(numNodes, 1);
        for (unsigned component = 0; component < numNodes; component++) {
            newNode(component) = (i & (1 << component)) >> component;
        }
        ret.push_back(newNode);
    }
    return ret;
}

int main(void) {
    static const unsigned numVisibleNodes = 3;
    static const unsigned numHiddenNodes = 2;
    static const unsigned numInputData = 3;
    Matrix input(numVisibleNodes, numInputData);

    {
        input(0, 0) = 1;
        input(1, 0) = 0;
        input(2, 0) = 0;

        input(0, 1) = 0;
        input(1, 1) = 1;
        input(2, 1) = 0;

        input(0, 1) = 0;
        input(1, 1) = 0;
        input(2, 1) = 1;
    }

    RBM rbm(numVisibleNodes, numHiddenNodes);
    for (int iteration = 0; iteration < 4096; iteration++) {
        RBM::UpdatedParams updateParams = rbm.getUpdateParamsByCD(input);
        rbm.updateByCD(updateParams);
    }

    // for debugging
    // power set of visible nodes
    std::vector<Matrix> allVisibleNodes = RBM::getPowerSet(numVisibleNodes);

    // power set of hidden nodes
    std::vector<Matrix> allHiddenNodes = RBM::getPowerSet(numHiddenNodes);

    std::vector<double> energies;
    double Z = 0;
    for (const auto& visibleNode : allVisibleNodes) {
        double sum = 0;
        for (const auto& hiddenNode : allHiddenNodes) {
            sum += rbm.getEnergy(visibleNode, hiddenNode);
        }
        Z += sum;
        energies.push_back(sum);
    }
    for (int i = 0; i < 8; i++) {
        std::cout <<
            ((i & (1 << 0)) >> 0) <<
            ((i & (1 << 1)) >> 1) <<
            ((i & (1 << 2)) >> 2) <<
            " " <<
            std::setprecision(3) << std::fixed <<
            (energies.at(i) / Z) << std::endl;
    }
    return 0;
}
