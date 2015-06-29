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
            }
        }
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
    Matrix operator()(const Matrix& input) {
        Matrix output(input.getRow(), input.getCol());
        for (int row = 0; row < input.getRow(); row++) {
            for (int col = 0; col < input.getCol(); col++) {
                output(row, col) =  Parent::operator()(3);
            }
        }
        return output;
    };
};

int main(void){
    struct Layer {
        const int numNode;
    };

    const Layer prevLayer{2};
    const Layer nextLayer{2};
    static const int numBatch = 3;

    // weight
    Matrix w(numNode, numBatch);
    // bias
    Matrix b(numBatch, 1);
    // input
    Matrix x(numNode, numBatch);
    // activation function
    ActivationFunction f([](double input) {
        return std::max<double>(input, 0);
    });

    Matrix u = Matrix::Mult(w, x);
    Matrix z = f(u);
    return 0;
}
