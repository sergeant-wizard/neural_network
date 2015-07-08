#pragma once

#include <string>

class MnistLoader {
public:
    MnistLoader(const std::string& fileName);
    virtual ~MnistLoader();
    uint8_t getPixel(int pixelIndex) const;
    uint8_t getPixel(int imageIndex, int row, int col) const;

private:
    static uint32_t readFlippedEndian(std::ifstream* fin);
    static uint32_t convertEndian(uint32_t source);
    uint32_t pixelIndex(int imageIndex, int row, int col) const;
    uint32_t magicNumber;
    uint32_t numImages;
    uint32_t row;
    uint32_t col;
    uint32_t numPixels;
    uint8_t* pixels;
};
