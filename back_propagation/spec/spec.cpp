#include "spec.h"
#include "../matrix.h"

bool testMatrix() {
    Spec spec;

    spec.addTest([]() {
        // assignment
        Matrix a(2, 3);
        a(1, 2) = 3;
        Matrix c = a;
        return a == c;
    });
    spec.addTest([]() {
        // swap
        Matrix a(2, 3);
        Matrix b(3, 2);
        Matrix c = b;
        Matrix::swap(a, b);
        return a == c;
    });
    spec.addTest([]() {
        // fill
        Matrix a(2, 2);
        a(0, 0) = 1;
        a(0, 1) = 1;
        a(1, 0) = 1;
        a(1, 1) = 1;
        Matrix b(2, 2);
        b.fill(1);
        return a == b;
    });
    spec.addTest([]() {
        // row, col
        Matrix a(2, 3);
        return a.getRow() == 2 && a.getCol() == 3;
    });
    spec.addTest([]() {
        // operator()(int), operator()(int, int)
        Matrix a(2, 3);
        a(1, 1) = 8;
        return a(4) == 8;
    });
    spec.addTest([]() {
        // component product
        Matrix a(2, 2);
        a(0, 0) = 1; a(0, 1) = 2;
        a(1, 0) = 2; a(1, 1) = 3;

        Matrix b(2, 2);
        b(0, 0) = 10; b(0, 1) = 30;
        b(1, 0) = 20; b(1, 1) = 40;

        Matrix c(2, 2);
        c(0, 0) = 10; c(0, 1) = 60;
        c(1, 0) = 40; c(1, 1) = 120;
        return Matrix::ComponentProduct(a, b) == c;
    });
    spec.addTest([]() {
        // mult
        Matrix a(2, 2);
        a(0, 0) =  1; a(0, 1) = 2;
        a(1, 0) = -1; a(1, 1) = 3;

        Matrix b(2, 2);
        b(0, 0) = 10; b(0, 1) = 30;
        b(1, 0) = 20; b(1, 1) = 40;

        Matrix c(2, 2);
        c(0, 0) = 50; c(0, 1) = 110;
        c(1, 0) = 50; c(1, 1) =  90;
        return Matrix::Mult(a, b) == c;
    });
    spec.addTest([]() {
        // multT1
        Matrix a(2, 2);
        a(0, 0) =  1; a(0, 1) = 2;
        a(1, 0) = -1; a(1, 1) = 3;

        Matrix b(2, 2);
        b(0, 0) = 10; b(0, 1) = 30;
        b(1, 0) = 20; b(1, 1) = 40;

        Matrix c(2, 2);
        c(0, 0) = -10; c(0, 1) = -10;
        c(1, 0) =  80; c(1, 1) = 180;
        return Matrix::MultT1(a, b) == c;
    });
    spec.addTest([]() {
        // multT2
        Matrix a(2, 2);
        a(0, 0) =  1; a(0, 1) = 2;
        a(1, 0) = -1; a(1, 1) = 3;

        Matrix b(2, 2);
        b(0, 0) = 10; b(0, 1) = 30;
        b(1, 0) = 20; b(1, 1) = 40;

        Matrix c(2, 2);
        c(0, 0) =  70; c(0, 1) = 100;
        c(1, 0) =  80; c(1, 1) = 100;
        return Matrix::MultT2(a, b) == c;
    });
    spec.addTest([]() {
        // operator -=
        Matrix a(2, 2);
        a(0, 0) =  1; a(0, 1) = 2;
        a(1, 0) = -1; a(1, 1) = 3;

        Matrix b(2, 2);
        b(0, 0) = 10; b(0, 1) = 30;
        b(1, 0) = 20; b(1, 1) = 40;

        Matrix c(2, 2);
        c(0, 0) =  -9; c(0, 1) = -28;
        c(1, 0) = -21; c(1, 1) = -37;
        return (a -= b) == c;
    });
    spec.addTest([]() {
        // operator *=
        Matrix a(2, 2);
        a(0, 0) =  1; a(0, 1) = 2;
        a(1, 0) = -1; a(1, 1) = 3;

        Matrix b(2, 2);
        b(0, 0) =  10; b(0, 1) = 20;
        b(1, 0) = -10; b(1, 1) = 30;

        return (a *= 10) == b;
    });
    spec.addTest([]() {
        // operator -
        Matrix a(2, 2);
        a(0, 0) =  1; a(0, 1) = 2;
        a(1, 0) = -1; a(1, 1) = 3;

        Matrix b(2, 2);
        b(0, 0) = 10; b(0, 1) = 30;
        b(1, 0) = 20; b(1, 1) = 40;

        Matrix c(2, 2);
        c(0, 0) =  -9; c(0, 1) = -28;
        c(1, 0) = -21; c(1, 1) = -37;
        return (a - b) == c;
    });
    spec.addTest([]() {
        // norm2
        Matrix a(2, 2);
        a(0, 0) =  1; a(0, 1) = 2;
        a(1, 0) = -3; a(1, 1) = 4;

        return a.norm2() == 30;
    });
    return spec.execTest();
}

int main(void) {
    bool result = true;
    result &= testMatrix();
    if (result) {
        std::cout << "all tests have succeeded" << std::endl;
        return 0;
    } else {
        std::cout << "at least one test failed" << std::endl;
        return 1;
    }
}
