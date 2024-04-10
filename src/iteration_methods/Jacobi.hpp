// #pragma once

// #include <iostream>
// #include "../CSR_matrix.hpp"
// #include "../Dense_matrix.hpp"

// template <typename T>
// std::vector<T> Jacobi(const Matrix<T> &A, const std::vector<T> &b, const std::vector <T> &x, T tolerance, int Nmax){
//     std::vector<T> x1 = x;
//     std::vector<T> r = A.multiply(x1)-b;
//     int n = 0;
//     while(mod(r) > tolerance and n < Nmax){
//         for (int i = 0; i < nonzero.size()-1; i++) {
// 			double d = 0;
// 			double sum = 0;
// 			for (int k = nonzero[i]; k < nonzero[i + 1]; k++) {
// 				if (i != columns[k]) {
// 					sum += elements[k] * res[columns[k]];
// 				}
// 				else {
// 					d = elements[k];
// 				}
// 			}
// 			res[i] = (b[i] - sum) / d;
// 		}
//         n++;
//     }
//     return x1;
// }
