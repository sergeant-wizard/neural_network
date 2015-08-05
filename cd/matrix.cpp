#include "matrix.h"
#include <iostream>
#include <random>

Matrix::Matrix(int row, int col):
    row(row),
    col(col),
    components(new double[row * col]())
{
}
Matrix::Matrix(const Matrix& other) :
    row(other.getRow()),
    col(other.getCol()),
    components(new double[row * col]())
{
    for (int i = 0; i < row * col; i++) {
        components[i] = other(i);
    }
}
Matrix& Matrix::operator=(const Matrix& other) {
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
void Matrix::swap(Matrix& left, Matrix& right) {
    std::swap(left.row, right.row);
    std::swap(left.col, right.col);
    std::swap(left.components, right.components);
}
Matrix::~Matrix() {
    if (components)
        delete[] components;
}
void Matrix::fill(double value) {
    for (int index = 0; index < row * col; index++) {
        components[index] = value;
    }
}
void Matrix::randomize(double min, double max) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(min, max);
    for (int index = 0; index < row * col; index++) {
        components[index] = distribution(generator);
    }
}
double Matrix::norm2() const {
    double ret = 0;
    for (int index = 0; index < row * col; index++) {
        ret += components[index] * components[index];
    }
    return ret;
}
int Matrix::getRow() const {
    return row;
}
int Matrix::getCol() const {
    return col;
}
const double& Matrix::operator()(int index) const {
    return components[index];
}
double& Matrix::operator()(int index) {
    return components[index];
}
double& Matrix::operator()(int row, int col) {
    return components[row * this->col + col];
}
const double& Matrix::operator()(int row, int col) const {
    return components[row * this->col + col];
}
double* Matrix::begin() {
    return &components[0];
}
double* Matrix::end() {
    return &components[row * col - 1];
}
void Matrix::print() const {
    for (int row = 0; row < getRow(); row++) {
        for (int col = 0; col < getCol(); col++) {
            std::cout << operator()(row, col) << " ";
        }
        std::cout << std::endl;
    }
}
Matrix Matrix::ComponentProduct(const Matrix& left, const Matrix& right) {
    Matrix ret = left;
    for (int index = 0; index < left.row * left.col; index++) {
        ret(index) *= right(index);
    }
    return ret;
}
Matrix Matrix::Mult(const Matrix& left, const Matrix& right) {
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
Matrix Matrix::MultT1(const Matrix& left, const Matrix& right) {
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
Matrix Matrix::MultT2(const Matrix& left, const Matrix& right) {
    // left * right^T
    Matrix ret(left.getRow(), right.getRow());
    for (int row = 0; row < ret.getRow(); row++) {
    for (int col = 0; col < ret.getCol(); col++) {
        double sum = 0;
        for (int sumIndex = 0; sumIndex < left.getCol(); sumIndex++) {
            sum += left(row, sumIndex) * right(col, sumIndex);
        }
        ret(row, col) = sum;
    }}
    return ret;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    for (int index = 0; index < row * col; index++) {
        components[index] += other.components[index];
    }
    return (*this);
}
Matrix& Matrix::operator-=(const Matrix& other) {
    for (int index = 0; index < row * col; index++) {
        components[index] -= other.components[index];
    }
    return (*this);
}
Matrix& Matrix::operator*=(const double coef) {
    for (int index = 0; index < row * col; index++) {
        components[index] *= coef;
    }
    return (*this);
}

Matrix operator*(double coef, const Matrix& right) {
    Matrix ret(right);
    for (int i = 0; i < right.row * right.col; i++) {
        ret(i) *= coef;
    }
    return ret;
}

Matrix operator+(const Matrix& left, const Matrix& right) {
    Matrix ret(left);
    for (int i = 0; i < right.row * right.col; i++) {
        ret(i) += right(i);
    }
    return ret;
}
Matrix operator-(const Matrix& left, const Matrix& right) {
    Matrix ret(left);
    for (int i = 0; i < right.row * right.col; i++) {
        ret(i) -= right(i);
    }
    return ret;
}

bool operator==(const Matrix& left, const Matrix& right) {
    if (left.row != right.row) {
        return false;
    }
    if (left.col != right.col) {
        return false;
    }
    for (int index = 0; index < left.row * left.col; index++) {
        if (left(index) != right(index)) {
            return false;
        }
    }
    return true;
}
