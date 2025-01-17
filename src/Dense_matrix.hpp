#pragma once

#include <vector>
#include <cmath>
#include<iostream>
template<typename T>

class Dense_matrix{
private:
    std::vector<T> M{};
    int row_length = 0;
public:
    Dense_matrix(const std::vector<T> &v, int length){
        this->M = v;
        this->row_length = length;
    }
    ~Dense_matrix() = default;
    const T operator() (int i, int j) const{
        return M[i*row_length + j];
    }
    const std::vector<T>& get_elements() const {return M;}
    int get_length() const {return row_length;}
    const void operator + (const Dense_matrix &A) {
        if (M.size() == (A.M).size()) {
            for (int i = 0; i < (A.M).size(); i++)
                M[i] += (A.M)[i];
        }
    }
    const void operator - (const Dense_matrix &A) {
        if (M.size() == (A.get_elements()).size()) {
            for (int i = 0; i < (A.M).size(); i++)
                M[i] -= (A.M)[i];
        }
    }

    const std::vector<T> get_column(int col_num) const{
        std::vector<T> column;
        for(int i = 0; i < M.size()/row_length; i++){
            column.push_back(M[i*row_length + col_num]);
        }
        return column;
    }
    const std::vector<T> get_row(int row_num) const{
        std::vector<T> row;
        for(int j = 0; j < row_length; j++){
            row.push_back(M[row_num*row_length + j]);
        }
        return row;
    }

    void swap_column(int col_num, const std::vector<T>& new_col) {
        for(int i = 0; i < M.size()/row_length; i++){
            M[i*row_length + col_num] = new_col[i];
        }
    }

    void swap_element(int i, int j, T num) {
        M[i * row_length + j] = num;
    }


};

template<typename T>
double operator * (const std::vector<T>& x, const std::vector<T>& y){
    double res = 0;
    for(int i = 0; i < x.size(); i++){
        res+= x[i]*y[i];
    }
    return res;
}

template<typename T>
std::vector<T> mul_components(const std::vector<T>& x, const std::vector<T>& y){
    std::vector<T> res;
    for(int i = 0; i < x.size(); i++){
        res.push_back(x[i]*y[i]);
    }
    return res;
}


template<typename T>
std::vector<T> transposed_prod(const std::vector<T>& x) {
    std::vector<T> res;
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x.size(); j++) {
            res.push_back(x[i] * x[j]);
        }
    }
    return res;
}


template<typename T, typename Scalar>
std::vector<T> operator * (const std::vector<T>&x, Scalar num){
    std::vector<T> res;
    for(int i = 0; i < x.size(); i++){
        res.push_back(x[i]*num);
    }
    return res;
}

template<typename T>
std::vector<T> operator*(const std::vector<T> &v, double n){
	std::vector<T> ret = std::vector<T>(v.size());
	for(size_t i = 0; i < v.size(); i++){
		ret[i] = n * v[i];
	}
	return ret;
}

template<typename T>
std::vector<T> operator*(double n, const std::vector<T> &v){
	return v * n;
}
template<typename T, typename Scalar>
std::vector<T> operator / (const std::vector<T>& x, Scalar num){
    std::vector<T> res;
    for(int i = 0; i < x.size(); i++){
        res.push_back(x[i]/num);
    }
    return res;
}

template<typename T>
std::vector<T> operator* (const Dense_matrix<T> &A, const std::vector<T>&x){
    std::vector<T> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x.size(); j++) {
            res[i] += (A.get_elements())[i * A.get_length() + j] * x.at(j);
        }
    }
    return res;
}

template<typename T>
Dense_matrix<T> operator * (const Dense_matrix<T> &A, const Dense_matrix<T> &B){
    std::vector<T> res;
    for(int i = 0; i < A.get_elements().size()/A.get_length(); i++){
        for(int j = 0; j < B.get_length(); j++) {
            res.push_back(A.get_row(i) * B.get_column(j));
        }
    }
    Dense_matrix<T> matrix(res, A.get_length());
    return matrix;
}

template<typename T>
bool operator == (const Dense_matrix<T> &A, const Dense_matrix<T> &B){
    for(int i = 0; i < A.get_elements().size(); i ++){
        if (std::abs(A.get_elements()[i] - B.get_elements()[i]) > 0.01)
            return false;
    }
    return true;
}

template<typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& v){
    for(int i = 0; i < v.size(); i++){
        os << v[i] << " ";
    }
    return os;
}

template<typename T>
std::vector<T> operator + (const std::vector<T>& x, const std::vector<T>& y){
    std::vector<T> res;
    for(int i = 0; i < x.size(); i++){
        res.push_back(x[i] + y[i]);
    }
    return res;
}

template<typename T>
std::vector<T> operator - (const std::vector<T>& x, const std::vector<T>& y){
    std::vector<T> res;
    for(int i = 0; i < x.size(); i++){
        res.push_back(x[i] - y[i]);
    }
    return res;
}

template <typename T>
double squares(const std::vector<T>& x){
    double res = 0;
    for(int i = 0; i < x.size(); i++){
        res+= pow(x[i],2);
    }
    return res;
}

template <typename T>
double mod(const std::vector<T>& x){
    double res = 0;
    for(int i = 0; i < x.size(); i++){
        res+= pow(x[i],2);
    }
    return pow(res,0.5);
}

template <typename T>
int sign(const std::vector<T>& x){
    if(x[0] < 0)
        return -1;
    return 1;
}

template <typename T>
void transpose(Dense_matrix<T> &M){
    for (int i = 0; i < M.get_elements().size()/M.get_length(); i++){
        for (int j = 0; j < M.get_length(); j++){
            M.swap_element(i,j,M(j,i));
        }
    }
}

template <typename T>
bool upper_diagonal(Dense_matrix<T> &M){
    for(int i = 0; i < M.get_elements().size()/M.get_length(); i++){
        for(int j = 0; j < i; j++){
            if (M(i,j) != 0)
                return false;
        }
    }
    return true;
}

template <typename T>
bool unitary(Dense_matrix<T> &M){
    for(int i = 0; i < M.get_elements().size()/M.get_length(); i++){
        if(std::abs(squares(M.get_row(i))-1) > 0.1)
            return false;
}
    return true;
}
