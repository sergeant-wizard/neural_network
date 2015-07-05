#include "png_generator.h"
#include <fstream>

PngWriter::PngWriter(int width, int height) :
    width(width),
    height(height),
    pixelIndex(0)
{
    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for(int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(sizeof(png_bytep) * width * bytePerPixel);
    }
}

PngWriter::~PngWriter() {
    for(int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
}

int PngWriter::read(const std::string& filename) {
    FILE *fp = fopen(filename.c_str(), "r");
    if(!fp)
        return -1;
    for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
        png_bytep pixel = &row_pointers[row][col * bytePerPixel];
    for (int colorIndex = 0; colorIndex < bytePerPixel; colorIndex++) {
        int tmp = fgetc(fp);
        pixel[colorIndex] = tmp;
    }}}
    fclose(fp);
    return 0;
}

int PngWriter::write(const std::string& filename) {
    FILE *fp = fopen(filename.c_str(), "wb");
    if(!fp)
        return -1;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png)
        return -1;

    png_infop info = png_create_info_struct(png);
    if (!info)
        return -1;

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    fclose(fp);
    return 0;
}

void PngWriter::addPixel(uint8_t grayScale) {
    int row = pixelIndex / (width);
    int col = pixelIndex % width;
    pixelIndex++;
    png_bytep pixel = &row_pointers[row][col * bytePerPixel];
    for (int colorIndex = 0; colorIndex < bytePerPixel - 1; colorIndex++) {
        pixel[colorIndex] = grayScale;
    }
    pixel[bytePerPixel - 1] = 255; // alpha
}

void PngWriter::createTestData(const std::string& filename) {
    unsigned char buffer[16 * 4];
    for (int i=0; i<16; i++) {
        buffer[i*4 + 0] = i * 16;
        buffer[i*4 + 1] = i * 16;
        buffer[i*4 + 2] = i * 16;
        buffer[i*4 + 3] = 255;
    }
    std::ofstream fout(filename);
    fout.write(reinterpret_cast<const char*>(buffer), 16*4);
}

int main(void) {
    static const int width = 28;
    static const int height = 28;

    PngWriter writer(width, height);
    writer.read("out.bin");
    writer.write("out.png");

    return 0;
}
