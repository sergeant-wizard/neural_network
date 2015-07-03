#include <fstream>
#include <iostream>

uint32_t convertEndian(uint32_t source) {
    return
        (source & 0xFF000000) >> 24 |
        (source & 0x00FF0000) >> 8 |
        (source & 0x0000FF00) << 8 |
        (source & 0x000000FF) << 24;
}

uint32_t readFlippedEndian(std::ifstream* fin) {
    uint32_t flipped = 0;
    fin->read(reinterpret_cast<char*>(&flipped), 4);
    return convertEndian(flipped);
}

int main(void) {
    std::ifstream fin("t10k-images-idx3-ubyte");
    uint32_t magicNumber = readFlippedEndian(&fin);
    uint32_t numImages = readFlippedEndian(&fin);
    uint32_t row = readFlippedEndian(&fin);
    uint32_t col = readFlippedEndian(&fin);

    std::ofstream fout("out.bin");
    for (unsigned i = 0; i < row * col; i++) {
        static const char alpha = 0xFF;
        char buffer;
        fin.read(&buffer, 1);
        fout.write(&buffer, 1);
        fout.write(&buffer, 1);
        fout.write(&buffer, 1);
        fout.write(&alpha, 1);
    }
}
