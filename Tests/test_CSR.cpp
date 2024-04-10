#include <iostream>
#include "../src/CSR_matrix.hpp"
#include "../src/Dense_matrix.hpp"
#include "gtest/gtest.h"
#include "../src/iteration_methods/MPI.hpp"
#include "../src/iteration_methods/MPI_chebyshev.hpp"
#include "../src/iteration_methods/sim_gauss_zeidel.hpp"
#include "../src/iteration_methods/gauss_seidel.hpp"
#include "../src/iteration_methods/Jacobi.hpp"
#include "../src/iteration_methods/fastest_gradient_descent.hpp"

TEST(csr, get_element){
	std::vector<int> v = {1, 2, 0, 3, 0, 0, 4, 0, 0, 1, 0, 11};
	size_t width = 4;
	Matrix<int> mtr = Matrix<int>(v, width);
	ASSERT_EQ(mtr(1, 2), 4);
	ASSERT_EQ(mtr(1, 1), 0);
	ASSERT_EQ(mtr(2, 3), 11);
}

TEST(csr, vector_multiply){
	std::vector<int> v = {1, 2, 0, 3, 0, 0, 4, 0, 0, 1, 0, 11};
	size_t width = 4;
	Matrix<int> mtr = Matrix<int>(v, width);
	std::vector<int> v1 = {1, 2, 3, 4};
	std::vector<int> res1 = {17, 12, 46};
	ASSERT_EQ(mtr.multiply(v1), res1);
	std::vector<int> v2 = {4, 3, 2, 1};
	std::vector<int> res2 = {13, 8, 14};
	ASSERT_EQ(mtr.multiply(v2), res2);
}

TEST(csr, simple_iter_method){
	std::vector<double> data = {5, -2, 0, -1, -2, 3, 0, -0.2, 0, 0, 7, 0, -1, -0.2, 0, 2};
	Matrix<double> mtr = Matrix<double>(data, 4);
	std::vector<double> v = {-4, 0, -2, 9};
	std::vector<double> start = {1, 1, 1, 1};
	std::vector<double> res = MPI(mtr, v, start, 0.0001, 1000, 0.25);
	std::vector<double> correct = {0.37555555555555556, 0.56666666666666667, -0.28571428571428571, 4.74444444444444444};
	for(size_t i = 0; i < res.size(); i++){
		ASSERT_NEAR(res[i], correct[i], 0.0001);
	}
}

TEST(csr, chebyshev_SIM){
	std::vector<double> data = {5, -2, 0, -1, -2, 3, 0, -0.2, 0, 0, 7, 0, -1, -0.2, 0, 2};
	Matrix<double> mtr = Matrix<double>(data, 4);
	std::vector<double> v = {-4, 0, -2, 9};
	std::vector<double> start = {1, 1, 1, 1};
	std::vector<double> res = chebyshev_mpi(mtr, v, start, 0.0001, 8, 1000, 1.129, 7.0);
	std::vector<double> correct = {0.37555555555555556, 0.56666666666666667, -0.28571428571428571, 4.74444444444444444};
	for(size_t i = 0; i < res.size(); i++){
		ASSERT_NEAR(res[i], correct[i], 0.0001);
	}
}

// TEST(csr, jakobi_method){
// 	std::vector<double> data = {5, -2, 0, -1, -2, 3, 0, -0.2, 0, 0, 7, 0, -1, -0.2, 0, 2};
// 	Matrix<double> mtr = Matrix<double>(data, 4);
// 	std::vector<double> v = {-4, 0, -2, 9};
// 	std::vector<double> start = {1, 1, 1, 1};
// 	std::vector<double> res = Jacobi(mtr, v, start, 0.0001, 1000);
// 	std::vector<double> correct = {0.37555555555555556, 0.56666666666666667, -0.28571428571428571, 4.74444444444444444};
// 	for(size_t i = 0; i < res.size(); i++){
// 		ASSERT_NEAR(res[i], correct[i], 0.0001);
// 	}
// }

TEST(csr, gauss_seidel_method){
	std::vector<double> data = {5, -2, 0, -1, -2, 3, 0, -0.2, 0, 0, 7, 0, -1, -0.2, 0, 2};
	Matrix<double> mtr = Matrix<double>(data, 4);
	std::vector<double> v = {-4, 0, -2, 9};
	std::vector<double> start = {1, 1, 1, 1};
	std::vector<double> res = Gauss_seidel(mtr, v, start, 0.0001, 1000);
	std::vector<double> correct = {0.37555555555555556, 0.56666666666666667, -0.28571428571428571, 4.74444444444444444};
	for(size_t i = 0; i < res.size(); i++){
		ASSERT_NEAR(res[i], correct[i], 0.0001);
	}
}

TEST(csr, gradient_descent){
	std::vector<double> data = {5, -2, 0, -1, -2, 3, 0, -0.2, 0, 0, 7, 0, -1, -0.2, 0, 2};
	Matrix<double> mtr = Matrix<double>(data, 4);
	std::vector<double> v = {-4, 0, -2, 9};
	std::vector<double> start = {1, 1, 1, 1};
	std::vector<double> res = Gradient_descent(mtr, v, start, 0.0001, 1000);
	std::vector<double> correct = {0.37555555555555556, 0.56666666666666667, -0.28571428571428571, 4.74444444444444444};
	for(size_t i = 0; i < res.size(); i++){
		ASSERT_NEAR(res[i], correct[i], 0.0001);
	}
}

TEST(csr, sym_gauss_seidel_method){
	std::vector<double> data = {5, -2, 0, -1, -2, 3, 0, -0.2, 0, 0, 7, 0, -1, -0.2, 0, 2};
	Matrix<double> mtr = Matrix<double>(data, 4);
	std::vector<double> v = {-4, 0, -2, 9};
	std::vector<double> start = {1, 1, 1, 1};
	std::vector<double> res = sim_gauss_zeidel(mtr, v, start, 0.0001, 1000);
	std::vector<double> correct = {0.37555555555555556, 0.56666666666666667, -0.28571428571428571, 4.74444444444444444};
	for(size_t i = 0; i < res.size(); i++){
		ASSERT_NEAR(res[i], correct[i], 0.0001);
	}
}
