#pragma once
#include <iostream>
#include "../CSR_matrix.hpp"
#include "../Dense_matrix.hpp"

template <typename T>
std::vector<T> Gradient_descent(const Matrix<T> &A, const std::vector<T> &b,const std::vector<T> &x, T tolerance){
    T tau;
    std::vector<T> x1 = x;
    std::vector<T> r = A.multiply(x1) - b;
    int n = 0;
    while(mod(r) > tolerance){
        tau = r*r/ (r*A.multiply(r));
        x1 = x1 - tau*(A.multiply(x1) - b);
        r = A.multiply(x1) - b;
        file << x1 <<  std::endl;
        n++;
    }
    file.close();
        return x1;
}
