#include <fstream>
#include <iostream>

#include "mnist_loader.h"

MnistLoader::MnistLoader(const std::string& filename) {
    std::ifstream fin(filename);
    magicNumber = readFlippedEndian(&fin);
    numImages = readFlippedEndian(&fin);
    row = readFlippedEndian(&fin);
    col = readFlippedEndian(&fin);
    numPixels = numImages * row * col;
    pixels = new uint8_t[numPixels];
    for (unsigned i = 0; i < numPixels; i++) {
        char buffer;
        fin.read(&buffer, 1);
        pixels[i] = static_cast<uint8_t>(buffer);
    }
}

MnistLoader::~MnistLoader() {
    delete[] pixels;
}

uint8_t MnistLoader::getPixel(int pixelIndex) const {
    return pixels[pixelIndex];
}

uint8_t MnistLoader::getPixel(int imageIndex, int row, int col) const {
    return pixels[pixelIndex(imageIndex, row, col)];
}

uint32_t MnistLoader::convertEndian(uint32_t source) {
    return
        (source & 0xFF000000) >> 24 |
        (source & 0x00FF0000) >> 8 |
        (source & 0x0000FF00) << 8 |
        (source & 0x000000FF) << 24;
}

uint32_t MnistLoader::pixelIndex(int imageIndex, int row, int col) const {
    return imageIndex * this->row * this->col + row * this->col + col;
}

uint32_t MnistLoader::readFlippedEndian(std::ifstream* fin) {
    uint32_t flipped = 0;
    fin->read(reinterpret_cast<char*>(&flipped), 4);
    return convertEndian(flipped);
}

// for testing
/*
int main(void) {
    MnistLoader mnistLoader("../mnist_loader/t10k-images-idx3-ubyte");
    for (int i = 0; i < 28; i++) {
        std::cout << static_cast<int>(mnistLoader.getPixel(0, 7, i)) << std::endl;
    }
}
*/
