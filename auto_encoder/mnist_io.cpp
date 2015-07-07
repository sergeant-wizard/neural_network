#include "mnist_io.h"
#include "png_generator.h"
#include "mnist_loader.h"
#include <math.h>
#include <assert.h>

namespace MnistIO {
void MatrixToPNG(const Matrix& matrix) {
    static const double max = static_cast<double>(std::numeric_limits<uint8_t>::max());
    PngWriter writer(numRows, numCols);
    for (int row = 0; row < matrix.getRow(); row++) {
        uint8_t byte = static_cast<uint8_t>(round(matrix(row) * max));
        writer.addPixel(byte);
    }
    writer.write("out.png");
}
void MatrixToPNG(const Matrix& matrix, int sampleIndex) {
    static const double max = static_cast<double>(std::numeric_limits<uint8_t>::max());
    PngWriter writer(numRows, numCols);
    for (int row = 0; row < matrix.getRow(); row++) {
        uint8_t byte = static_cast<uint8_t>(round(matrix(row, sampleIndex) * max));
        writer.addPixel(byte);
    }
    writer.write("out.png");
}
std::vector<Matrix> MnistToMatrix(const std::string& filename) {
    static const double max = static_cast<double>(std::numeric_limits<uint8_t>::max());
    MnistLoader mnistLoader(filename);
    std::vector<Matrix> ret(numImages, Matrix(numRows * numCols, 1));
    for (int imageIndex = 0; imageIndex < numImages; imageIndex++) {
    for (int pixelIndex = 0; pixelIndex < numRows * numCols; pixelIndex++) {
        ret.at(imageIndex)(pixelIndex) = static_cast<double>(mnistLoader.getPixel(imageIndex * numRows * numCols + pixelIndex)) / max;
    }}
    return ret;
}
std::vector<Matrix> MnistToMatrixInBatches(const std::string& filename, int numSamples) {
    static const double max = static_cast<double>(std::numeric_limits<uint8_t>::max());
    MnistLoader mnistLoader(filename);
    assert(numImages % numSamples == 0);
    const int numBatches = numImages / numSamples;
    std::vector<Matrix> ret(numBatches, Matrix(numRows * numCols, numSamples));
    int pixelIndexAccumulator = 0;
    for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
    for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
    for (int pixelIndex = 0; pixelIndex < numRows * numCols; pixelIndex++) {
        ret.at(batchIndex)(pixelIndex, sampleIndex) = static_cast<double>(mnistLoader.getPixel(pixelIndexAccumulator++)) / max;
    }}}
    return ret;
}
}

// for testing
/*
int main(void) {
    std::vector<Matrix> mat = MnistIO::MnistToMatrixInBatches("../mnist_loader/t10k-images-idx3-ubyte", 100);
    MnistIO::MatrixToPNG(mat.at(99), 99);
    return 0;
}
*/
