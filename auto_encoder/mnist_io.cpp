#include "mnist_io.h"
#include "png_generator.h"
#include "mnist_loader.h"
#include <math.h>

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
}

// for testing
int main(void) {
    std::vector<Matrix> mat = MnistIO::MnistToMatrix("../mnist_loader/t10k-images-idx3-ubyte");
    MnistIO::MatrixToPNG(mat.at(3));
    return 0;
}
