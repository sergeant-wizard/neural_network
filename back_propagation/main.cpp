#include<vector>
#include<functional>

class Matrix {
public:
    Matrix(int row, int col):
        row(row),
        col(col),
        components(new double[row * col])
    {}
    virtual ~Matrix() {
        delete[] components;
    }
    int getRow() const {
        return row;
    }
    int getCol() const {
        return col;
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

class ActivationFunction : public std::function<double(double)>
{
    using Parent = std::function<double(double)>;
public:
    ActivationFunction(const ActivationFunction& other) :
        Parent(other)
    {}
    ActivationFunction(const Parent& other) :
        Parent(other)
    {}
    Matrix operator()(const Matrix& input) const {
        Matrix output(input.getRow(), input.getCol());
        for (int row = 0; row < input.getRow(); row++) {
        for (int col = 0; col < input.getCol(); col++) {
                output(row, col) =  Parent::operator()(3);
            }}
        return output;
    };
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
        w(numNodes, prevLayer->getNumNodes()),
        b(numBatch, 1)
    {
    }
    int getNumNodes() const {
        return numNodes;
    }
    Matrix forwardPropagation(const Matrix& input) const {
        // input : numNodes * numBatch
        // u : nextLayer->getNumNodes() * 1
        Matrix u = Matrix::Mult(w, input);
        for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
        for (int batchIndex = 0; batchIndex < numBatch; batchIndex++) {
            u(nodeIndex, batchIndex) += b(numBatch, 1);
        }}
        return activationFunction(u);
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
        Layer(numBatch, numNodes, nullptr, ActivationFunction(nullptr))
    {
    }
    Matrix forwardPropagation(const Matrix& input) const {
        // identity mapping
        return input;
    }
};

int main(void){
    return 0;
}
