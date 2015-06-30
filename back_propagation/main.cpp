#include<vector>
#include<functional>
#include<iostream>
#include<algorithm>

/*
 * w : weight
 * b : bias
 * u : input
 * y : output
 * d : target output
 */

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
    Matrix& operator=(const Matrix& other) {
        if (components)
            delete[] components;
        row = other.row;
        col = other.col;
        components = new double[other.row * other.col]();
        for (int i = 0; i < row * col; i++) {
            components[i] = other(i);
        }
        return (*this);
    }
    static void swap(Matrix& left, Matrix& right) {
        std::swap(left.row, right.row);
        std::swap(left.col, right.col);
        std::swap(left.components, right.components);
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
    static Matrix ComponentProduct(const Matrix& left, const Matrix& right) {
        Matrix ret = left;
        for (int index = 0; index < left.row * left.col; index++) {
            ret(index) *= right(index);
        }
        return ret;
    }
    static Matrix Mult(const Matrix& left, const Matrix& right) {
        Matrix ret(left.getRow(), right.getCol());
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
    static Matrix MultT(const Matrix& left, const Matrix& right) {
        // left^T * right
        Matrix ret(left.getCol(), right.getCol());
        for (int row = 0; row < ret.getRow(); row++) {
        for (int col = 0; col < ret.getCol(); col++) {
            double sum = 0;
            for (int sumIndex = 0; sumIndex < left.getRow(); sumIndex++) {
                sum += left(sumIndex, row) * right(sumIndex, col);
            }
            ret(row, col) = sum;
        }}
        return ret;
    }
    friend Matrix operator-(const Matrix& left, const Matrix& right);

private:
    int row;
    int col;
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
        const ActivationFunction& activationFunction,
        const Layer* prevLayer) :
        numBatch(numBatch),
        numNodes(numNodes),
        activationFunction(activationFunction),
        prevLayer(prevLayer),
        w(numNodes, prevLayer ? prevLayer->getNumNodes() : 1),
        b(numBatch, 1),
        u(1, 1),
        delta(1, 1)
    {
    }
    int getNumNodes() const {
        return numNodes;
    }
    virtual Matrix forwardPropagation(const Matrix& input) {
        // input : numNodes * numBatch
        // u : nextLayer->getNumNodes() * 1
        Matrix u = Matrix::Mult(w, input);
        for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
        for (int batchIndex = 0; batchIndex < numBatch; batchIndex++) {
            u(nodeIndex, batchIndex) += b(numBatch, 0);
        }}
        Matrix::swap(u, this->u);
        return activationFunction.applyPrimaryFunction(this->u);
    }
    static void backwardPropagation(Layer& prevLayer, const Layer& nextLayer) {
        Matrix delta = Matrix::ComponentProduct(
            prevLayer.activationFunction.applyDerivativeFunction(prevLayer.u),
            Matrix::MultT(nextLayer.w, nextLayer.delta));
        Matrix::swap(prevLayer.delta, delta);
    }
    void print() const {
        std::cout << "weight" << std::endl;
        w.print();
        std::cout << "bias" << std::endl;
        b.print();
    }

protected:
    const int numBatch;
    const int numNodes;
    const ActivationFunction activationFunction;
    const Layer* prevLayer;
    // weight
    const Matrix w;
    // bias
    const Matrix b;
    Matrix u;
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
        return u = input;
    }
};

class LastLayer : public Layer {
public:
    using Layer::Layer;
    void setDelta(Matrix&& delta) {
        Matrix::swap(this->delta, delta);
    }
};

Matrix operator-(const Matrix& left, const Matrix& right) {
    Matrix ret(left);
    for (int i = 0; i < right.row * right.col; i++) {
        ret(i) -= right(i);
    }
    return ret;
}

int main(void) {
    // forward propagation
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

    FirstLayer firstLayer(numBatch, firstNodeNum, activationFunction);
    LastLayer secondLayer(numBatch, firstNodeNum, activationFunction, &firstLayer);

    Matrix input(firstNodeNum, numBatch);
    input(0, 0) = 1;
    input(1, 0) = 0;
    input(0, 1) = 0;
    input(1, 1) = 1;
    input(0, 2) = 1;
    input(1, 2) = 1;

    Matrix Y = secondLayer.forwardPropagation(firstLayer.forwardPropagation(input));

    // backward propagation

    // target output
    Matrix target(firstNodeNum, numBatch);
    target(0, 0) = 0;
    target(1, 0) = 0;
    target(0, 1) = 0;
    target(1, 1) = 0;
    target(0, 2) = 1;
    target(1, 2) = 0;

    secondLayer.setDelta(target - Y);
    Layer::backwardPropagation(firstLayer, secondLayer);

    return 0;
}
