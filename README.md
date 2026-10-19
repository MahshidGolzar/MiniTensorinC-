# mini_tensor

A tiny, from-scratch 2D matrix ("tensor") class in C++: no libraries beyond the standard library, no build system beyond a single compiler command.

## Why this exists

Every deep learning framework's `Tensor` type is, underneath, a flat buffer of numbers plus a shape describing how to read that buffer, with arithmetic operators defined on top. This is a small, honest version of that same idea — simple enough to explain line by line, and a useful stepping stone to reading real `torch::Tensor` code (see the `pytorch-parametric-swish` project).

## What's in here

- A `Matrix` class: fixed at construction to a given (rows, cols) shape, backed by a single flat `std::vector<double>`.
- `operator+` — elementwise addition.
- `elementwise_multiply` — elementwise (Hadamard) product, explicitly named to avoid confusion with matrix multiplication.
- `matmul` — real matrix multiplication (`(m×k) @ (k×n) → (m×n)`), implemented as three nested loops, the direct definition rather than any optimised algorithm.
- `main()` demonstrates all of the above on small hand-written matrices, with `assert()` checks against by-hand-computed expected values.

## Known limitations

- 2D only — no general N-dimensional support.
- `matmul` is the textbook triple-nested-loop definition — O(n³), no blocking/tiling/SIMD. Fine for small matrices, not how a real BLAS library would do it.
- No bounds-checking on `at(i, j)` — passing an out-of-range index is undefined behaviour, same as `std::vector::operator[]`. A production version would likely assert or throw on out-of-range access.

## Building and running

```bash
g++ -std=c++17 -Wall mini_tensor.cpp -o mini_tensor
./mini_tensor
```

(Built and tested with MSVC on Windows: `cl /EHsc /std:c++17 /W4 mini_tensor.cpp`.)
