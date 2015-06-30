#include<vector>
#include<functional>
#include<iostream>

class Matrix {
public:
    Matrix(int row, int col):
        row(row),
        col(col),
        components(new double[row * col]())
    {
    }
    Matrix(const Matrix& other) :
        row(other.getRow()),
        col(other.getCol()),
        components(new double[row * col]())
    {
        for (int i = 0; i < row * col; i++) {
            components[i] = other(i);
        }
    }
    virtual ~Matrix() {
        if (components)
            delete[] components;
    }
    int getRow() const {
        return row;
    }
    int getCol() const {
        return col;
    }
    const double& operator()(int index) const {
        return components[index];
    }
    double& operator()(int index) {
        return components[index];
    }
    double& operator()(int row, int col) {
        return components[row * this->col + col];
    }
    const double& operator()(int row, int col) const {
        return components[row * this->col + col];
    }
    double* begin() {
        return &components[0];
    }
    double* end() {
        return &components[row * col - 1];
    }
    void print() const {
        for (int row = 0; row < getRow(); row++) {
            for (int col = 0; col < getCol(); col++) {
                std::cout << operator()(row, col) << " ";
            }
            std::cout << std::endl;
        }
    }
    static Matrix Mult(const Matrix& left, const Matrix& right) {
        Matrix ret(left.row, right.col);
        for (int row = 0; row < ret.getRow(); row++) {
        for (int col = 0; col < ret.getCol(); col++) {
            double sum = 0;
            for (int sumIndex = 0; sumIndex < left.getCol(); sumIndex++) {
                sum += left(row, sumIndex) * right(sumIndex, col);
            }
            ret(row, col) = sum;
        }}
        return ret;
    }

private:
    const int row;
    const int col;
    double* components;
};

class MatrixFunction : public std::function<double(double)>
{
    using Parent = std::function<double(double)>;
public:
    MatrixFunction(const MatrixFunction& other) :
        Parent(other)
    {}
    MatrixFunction(const Parent& other) :
        Parent(other)
    {}
    Matrix operator()(const Matrix& input) const {
        Matrix output(input.getRow(), input.getCol());
        for (int row = 0; row < input.getRow(); row++) {
        for (int col = 0; col < input.getCol(); col++) {
                output(row, col) =  Parent::operator()(input(row, col));
            }}
        return output;
    };
};

class ActivationFunction {
public:
    ActivationFunction(const MatrixFunction& primaryFunction, const MatrixFunction& derivativeFunction) :
        primaryFunction(primaryFunction),
        derivativeFunction(derivativeFunction)
    {
    }
    Matrix applyPrimaryFunction(const Matrix& input) const {
        return primaryFunction(input);
    }
    Matrix applyDerivativeFunction(const Matrix& input) const {
        return derivativeFunction(input);
    }

private:
    const MatrixFunction primaryFunction;
    const MatrixFunction derivativeFunction;
};

class Layer {
public:
    Layer(
        const int numBatch,
        const int numNodes,
        const Layer* prevLayer,
        const ActivationFunction& activationFunction) :
        numBatch(numBatch),
        numNodes(numNodes),
        prevLayer(prevLayer),
        activationFunction(activationFunction),
        w(numNodes, prevLayer ? prevLayer->getNumNodes() : 1),
        b(numBatch, 1)
    {
    }
    int getNumNodes() const {
        return numNodes;
    }
    virtual Matrix forwardPropagation(const Matrix& input) const {
        // input : numNodes * numBatch
        // u : nextLayer->getNumNodes() * 1
        Matrix u = Matrix::Mult(w, input);
        for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
        for (int batchIndex = 0; batchIndex < numBatch; batchIndex++) {
            u(nodeIndex, batchIndex) += b(numBatch, 0);
        }}
        return activationFunction.applyPrimaryFunction(u);
    }

protected:
    const int numBatch;
    const int numNodes;
    const Layer* prevLayer;
    const ActivationFunction activationFunction;
    // weight
    const Matrix w;
    // bias
    const Matrix b;
};

class FirstLayer : public Layer {
public:
    FirstLayer(int numBatch, int numNodes) :
        Layer(
            numBatch,
            numNodes,
            nullptr,
            ActivationFunction(
                MatrixFunction(nullptr),
                MatrixFunction(nullptr)))
    {
    }
    Matrix forwardPropagation(const Matrix& input) const override {
        // identity mapping
        return input;
    }
};

int main(void){
    const int numBatch = 3;
    const int firstNodeNum = 2;
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
    FirstLayer firstLayer(numBatch, firstNodeNum);
    Layer secondLayer(numBatch, firstNodeNum, &firstLayer, activationFunction);

    Matrix input(firstNodeNum, numBatch);
    input(0, 0) = 1;
    input(1, 0) = 0;
    input(0, 1) = 0;
    input(1, 1) = 1;
    input(0, 2) = 1;
    input(1, 2) = 1;

    Matrix Y = secondLayer.forwardPropagation(firstLayer.forwardPropagation(input));

    // from forward propagation
    // to backward propagation

    // target output
    Matrix target(firstNodeNum, numBatch);
    target(0, 0) = 0;
    target(1, 0) = 0;
    target(0, 1) = 0;
    target(1, 1) = 0;
    target(0, 2) = 1;
    target(1, 2) = 0;

    return 0;
}
