#pragma once
#include <string>
#include <png.h>

class PngWriter {
public:
    PngWriter(int width, int height);
    virtual ~PngWriter();

    int read(const std::string& filename);
    int write(const std::string& filename);
    void addPixel(uint8_t grayScale);
    static void createTestData(const std::string& filename);

private:
    static const int bytePerPixel = 4;
    png_bytep *row_pointers;
    const int width;
    const int height;
    int pixelIndex;
};

