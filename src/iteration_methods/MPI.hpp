#pragma once

#include <iostream>
#include "../CSR_matrix.hpp"
#include "../Dense_matrix.hpp"

template <typename T>
std::vector<T> MPI(const Matrix<T> &A, const std::vector<T> &b,const std::vector<T> &x, T tolerance, int Nmax, T tau){
    std::vector<T> x1 = x;
    std::vector<T> r = A.multiply(x1) - b;
    int n = 0;
    while(mod(r) > tolerance and n < Nmax){
        x1 = x1 - tau*r;
        r = A.multiply(x1) - b;
        n++;
    }
    return x1;
}
