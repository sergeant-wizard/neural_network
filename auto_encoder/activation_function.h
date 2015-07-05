#pragma once
#include <functional>
#include "matrix.h"

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

