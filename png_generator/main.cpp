#include <stdlib.h>
#include <png.h>
#include <string>
#include <stdio.h>
#include <fstream>
#include <iostream>

class PngWriter {
public:
    PngWriter(int width, int height) :
        width(width),
        height(height)
    {
        row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
        for(int y = 0; y < height; y++) {
            row_pointers[y] = (png_byte*)malloc(sizeof(png_bytep) * width * bytePerPixel);
        }
    }

    virtual ~PngWriter() {
        for(int y = 0; y < height; y++) {
            free(row_pointers[y]);
        }
        free(row_pointers);
    }

    int read(const std::string& filename) {
        FILE *fp = fopen(filename.c_str(), "r");
        if(!fp)
            return -1;
        for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            png_bytep pixel = &row_pointers[row][col * bytePerPixel];
        for (int colorIndex = 0; colorIndex < bytePerPixel; colorIndex++) {
            int tmp = fgetc(fp);
            pixel[colorIndex] = tmp;
            std::cout << tmp << std::endl;
        }}}
        fclose(fp);
        return 0;
    }
    int write(const std::string& filename) {
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
    static void createTestData(const std::string& filename) {
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

private:
    static const int bytePerPixel = 4;
    png_bytep *row_pointers;
    const int width;
    const int height;
};

int main(int, char** argv) {
    static const int width = 4;
    static const int height = 4;

    PngWriter writer(width, height);
    writer.read("out.bin");
    writer.write("out.png");

    return 0;
}
