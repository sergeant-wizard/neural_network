#pragma once
#include<functional>
#include<vector>
#include<iostream>

class Spec {
public:
    using TestFunction = std::function<bool()>;
    void addTest(const TestFunction& testFunction) {
        tests.push_back(testFunction);
    }
    bool execTest() {
        bool ret = true;
        std::vector<int> failedTests;
        for (auto test = tests.begin(); test != tests.end(); ++test) {
            if (!(*test)()) {
                std::cout << "test at " << (test - tests.begin()) << " failed" << std::endl;
                ret = false;
            }
        }
        return ret;
    }

private:
    std::vector<TestFunction> tests;
};
