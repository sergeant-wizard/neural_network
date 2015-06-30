#pragma once

class Matrix {
public:
    Matrix(int row, int col);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    static void swap(Matrix& left, Matrix& right);
    virtual ~Matrix();
    void fill(double value);
    int getRow() const;
    int getCol() const;
    const double& operator()(int index) const;
    double& operator()(int index);
    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;
    double* begin();
    double* end();
    void print() const;
    static Matrix ComponentProduct(const Matrix& left, const Matrix& right);
    static Matrix Mult(const Matrix& left, const Matrix& right);
    static Matrix MultT1(const Matrix& left, const Matrix& right);
    static Matrix MultT2(const Matrix& left, const Matrix& right);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(const double coef);
    friend Matrix operator-(const Matrix& left, const Matrix& right);

private:
    int row;
    int col;
    double* components;
};

