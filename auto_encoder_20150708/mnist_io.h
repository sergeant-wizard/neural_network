#pragma once

#include <string>
#include <vector>
#include "matrix.h"

namespace MnistIO {
    void MatrixToPNG(const Matrix& matrix);
    void MatrixToPNG(const Matrix& matrix, int sampleIndex);
    std::vector<Matrix> MnistToMatrix(const std::string& filename);
    std::vector<Matrix> MnistToMatrixInBatches(const std::string& filename, int numSamples);
    static const int numRows = 28;
    static const int numCols = 28;
    static const int numImages = 10000;
}
