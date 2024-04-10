#pragma once

#include <iostream>
#include "../CSR_matrix.hpp"
#include "../Dense_matrix.hpp"

template <typename T>
std::vector<T> Gradient_descent(const Matrix<T> &A, const std::vector<T> &b,const std::vector<T> &x, T tolerance, int Nmax){
    T tau;
    std::vector<T> x1 = x;
    std::vector<T> r = A.multiply(x1) - b;
    int n = 0;
    while(mod(r) > tolerance and n < Nmax){
        tau = r*r/ (r*A.multiply(r));
        x1 = x1 - tau*(A.multiply(x1) - b);
        r = A.multiply(x1) - b;
        n++;
    }
    return x1;
}
