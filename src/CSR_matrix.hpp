#pragma once

#include <iostream>
#include <utility>
#include <map>
#include <vector>
#include <algorithm>


template<typename T>
class Matrix{
private:
    std::vector<T> values;
    std::vector<int> columns;
    std::vector<int> rows;
public:
    Matrix(const std::vector<T> &data, int width) : rows(data.size() / width + 1){
	int len = 0;
	for(T value : data){
		if(value != 0){
			len++;
		}
	}
	this->values.reserve(len);
	this->columns.reserve(len);
	this->rows[0] = 0;
	int count = 0;
	for(int i = 0; i < data.size(); i++){
		if(data[i] != 0){
			this->values.push_back(data[i]);
			this->columns.push_back(i % width);
			count++;
		}
		if(i % width == width - 1){
			this->rows[i / width + 1] = count;
		}
	}
}

    ~Matrix() = default;
    const std::vector<T>& get_values() const {return values;}
    const std::vector<int>& get_columns() const {return columns;}
    const std::vector<int>& get_rows() const {return rows;}
    const T operator() (int i, int j) const{
        for(int r = rows[i]; r < rows[i+1]; r++){
            if (columns[r] == j){
                return values[r];
            }
        }
        return 0;
    }
    std::vector<T> multiply(const std::vector<T> &x) const{
        std::vector<T> res(rows.size()-1, 0);
        for (int i = 0; i < x.size(); i++) {
            for (int j = rows[i]; j < rows[i + 1]; j++)
                res[i] += values[j] * x[columns[j]];
        }
        return res;
    }

    std::vector<T> multiply_LU(const std::vector<T> &x) const{
        std::vector<T> res(x.size(), 0);
        for (int i = 0; i < x.size(); i++) {
            for (int j = rows[i]; j < rows[i + 1]; j++)
                if (i == columns[j])
                    continue;
                else
                    res[i] += values[j] * x[columns[j]];
        }
        return res;
    }

};

template <typename T>
std::vector<T> inverse_diagonal(const Matrix <T> &A, int dim1, int dim2){
    std::vector<T> diag;
    for(int i=0; i < dim1; i++){
        for (int j=0; j < dim2; j++){
            if(i==j){
                diag.push_back(1/A(i,j));
            }
        }
    }

    return diag;
}
