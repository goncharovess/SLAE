#pragma once

#include <iostream>
#include "../CSR_matrix.hpp"
#include "../Dense_matrix.hpp"

template <typename T>
std::vector<T> Gauss_seidel(const Matrix<T> &A, const std::vector<T> &b, const std::vector <T> &x, T tolerance, int Nmax){
    std::vector<T> x0 = x;
    T diag_el;
    std::vector<T> r = A.multiply(x)-b;
    int n = 0;
    while(mod(r) > tolerance and n < Nmax){
        for(int i = 0 ; i < x.size(); i++){
            x0[i] = b[i];
            for(int j = A.get_rows()[i]; j < A.get_rows()[i+1]; j++){
                if (A.get_columns()[j] == i){
                    diag_el = A(i,i);
                    continue;
                }
                x0[i] -= A.get_values()[j]*x0[A.get_columns()[j]];
            }
            x0[i] /= diag_el;
        }
        r = A.multiply(x0)-b;
        n++;
    }
    return x0;
}
