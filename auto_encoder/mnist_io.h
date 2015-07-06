#pragma once

#include <string>
#include <vector>
#include "matrix.h"

namespace MnistIO {
    void MatrixToPNG(const Matrix& matrix);
    std::vector<Matrix> MnistToMatrix(const std::string& filename);
    static const int numRows = 28;
    static const int numCols = 28;
    static const int numImages = 10000;
}
