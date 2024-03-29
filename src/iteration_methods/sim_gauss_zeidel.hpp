#pragma once

#include <iostream>
#include "../CSR_matrix.hpp"
#include "../Dense_matrix.hpp"

template <typename T>
std::vector<T> sim_gauss_zeidel_iteration(const Matrix<T> &A, const std::vector<T> &b, const std::vector <T> &x){
    std::vector<T> x0 = x;
    T t;
    T diag_el;

    for(int i = 0; i < x.size(); i++){ //0.5
        t = x0[i];
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

    for(int i = x.size()-1; i-- > 0; i){ //1
        t = x0[i];
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

    return x0;
}


template <typename T>
std::vector<T> sim_gauss_zeidel(const Matrix<T> &A, const std::vector<T> &b, const std::vector <T> &x, T tolerance){
    std::vector<T> x0 = x;
    std::vector<T> r = A.multiply(x0)-b;
    int n = 0;
    while(mod(r) > tolerance){
        x0 = SOR_iteration(A,b,x0);
        r = A.multiply(x0)-b;
        file << mod(r) << " " << n << std::endl;
        n++;
    }
    return x0;
}