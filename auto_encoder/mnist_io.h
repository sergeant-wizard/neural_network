#pragma once

#include <string>
#include "matrix.h"

namespace MnistIO {
    void MatrixToPNG(const Matrix& matrix);
    Matrix MnistToMatrix(const std::string& filename);
    static const int numRows = 28;
    static const int numCols = 28;
}
