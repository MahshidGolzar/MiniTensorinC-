// mini_tensor.cpp
//
// A tiny, from-scratch 2D matrix ("tensor") class: no libraries beyond the
// standard library, no build system beyond a single compiler command.
//
// Point of this file: every deep learning framework's Tensor is, underneath,
// a flat buffer of numbers plus a shape describing how to read that buffer,
// with arithmetic operators defined on top. This is a small version
// of that same idea - understanding it from the inside makes the real
// torch::Tensor code in parametric_swish.cpp less of a black box.
//
// Build and run:
//   g++ -std=c++17 -Wall mini_tensor.cpp -o mini_tensor && ./mini_tensor
// (or, with MSVC: cl /EHsc /std:c++17 mini_tensor.cpp)

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

class Matrix {
 public:
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

  // Non-const access: returns a reference, so callers can both read AND
  // write through it, e.g. `m.at(0, 1) = 5.0;`
  float &at(int i, int j) {
    return data_[i * cols_ + j];
  }

  // Const access: same indexing, but for read-only (const) Matrix objects.
  float at(int i, int j) const {
    return data_[i * cols_ + j];
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }

  void print() const {
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        std::cout << at(i, j);
        if (j + 1 < cols_) std::cout << " ";
      }
      std::cout << "\nfloat";
    }
  }

 private:
  int rows_, cols_;
  std::vector<float> data_;  // flat buffer: element (i, j) lives at i*cols_ + j
};

// Elementwise addition: a and b must be the same shape.
Matrix operator+(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::invalid_argument("operator+: shape mismatch");
  }
  Matrix result(a.rows(), a.cols());
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) {
      result.at(i, j) = a.at(i, j) + b.at(i, j);
    }
  }
  return result;
}

// Elementwise (Hadamard) product - NOT matrix multiplication. Same shape required.
Matrix elementwise_multiply(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::invalid_argument("elementwise_multiply: shape mismatch");
  }
  Matrix result(a.rows(), a.cols());
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) {
      result.at(i, j) = a.at(i, j) * b.at(i, j);
    }
  }
  return result;
}

// Real matrix multiplication: a is (m x k), b is (k x n), result is (m x n).
Matrix matmul(const Matrix &a, const Matrix &b) {
  if (a.cols() != b.rows()) {
    throw std::invalid_argument("matmul: a.cols() must equal b.rows()");
  }
  Matrix result(a.rows(), b.cols());
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < b.cols(); ++j) {
      float sum = 0.0;
      for (int k = 0; k < a.cols(); ++k) {
        sum += a.at(i, k) * b.at(k, j);
      }
      result.at(i, j) = sum;
    }
  }
  return result;
}

int main() {
  // --- Build two small 2x2 matrices by hand ---
  Matrix a(2, 2);
  a.at(0, 0) = 1;  a.at(0, 1) = 2;
  a.at(1, 0) = 3;  a.at(1, 1) = 4;

  Matrix b(2, 2);
  b.at(0, 0) = 5;  b.at(0, 1) = 6;
  b.at(1, 0) = 7;  b.at(1, 1) = 8;

  std::cout << "a =\n"; a.print();
  std::cout << "b =\n"; b.print();

  // --- Elementwise add ---
  Matrix sum = a + b;
  std::cout << "\na + b =\n"; sum.print();
  assert(sum.at(0, 0) == 6);
  assert(sum.at(1, 1) == 12);

  // --- Elementwise multiply ---
  Matrix prod = elementwise_multiply(a, b);
  std::cout << "\na .* b (elementwise) =\n"; prod.print();
  assert(prod.at(0, 0) == 5);   // 1 * 5
  assert(prod.at(1, 1) == 32);  // 4 * 8

  // --- Real matrix multiplication ---
  Matrix mm = matmul(a, b);
  std::cout << "\na @ b (matrix multiply) =\n"; mm.print();
  // By hand: row0 . col0 = 1*5 + 2*7 = 19 ; row0 . col1 = 1*6 + 2*8 = 22
  assert(mm.at(0, 0) == 19);
  assert(mm.at(0, 1) == 22);
  assert(mm.at(1, 0) == 43);  // 3*5 + 4*7
  assert(mm.at(1, 1) == 50);  // 3*6 + 4*8

  // --- A non-square example, to prove matmul handles different shapes ---
  Matrix c(2, 3);  // 2x3
  c.at(0, 0) = 1; c.at(0, 1) = 0; c.at(0, 2) = 2;
  c.at(1, 0) = -1; c.at(1, 1) = 3; c.at(1, 2) = 1;

  Matrix d(3, 2);  // 3x2
  d.at(0, 0) = 3; d.at(0, 1) = 1;
  d.at(1, 0) = 2; d.at(1, 1) = 1;
  d.at(2, 0) = 1; d.at(2, 1) = 0;

  Matrix cd = matmul(c, d);  // (2x3) @ (3x2) -> (2x2)
  std::cout << "\nc @ d (2x3 times 3x2) =\n"; cd.print();
  assert(cd.rows() == 2 && cd.cols() == 2);
  assert(cd.at(0, 0) == 5);   // 1*3 + 0*2 + 2*1
  assert(cd.at(1, 1) == 2);   // -1*1 + 3*1 + 1*0

  std::cout << "\nAll assertions passed.\n";
  return 0;
}
