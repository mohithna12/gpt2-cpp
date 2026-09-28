#pragma once
// Minimal float32 tensor: a shape plus a pointer into memory owned elsewhere
// (the checkpoint buffer for weights, or an activation arena later on).

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace gpt2 {

struct Tensor {
    float* data = nullptr;
    std::vector<int64_t> shape;

    int64_t ndim() const { return static_cast<int64_t>(shape.size()); }
    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
    int64_t dim(int64_t i) const { return shape[i < 0 ? i + ndim() : i]; }

    // Row pointer for a 2-D tensor (row-major, contiguous).
    float* row(int64_t r) { return data + r * shape[1]; }
    const float* row(int64_t r) const { return data + r * shape[1]; }

    bool has_shape(std::initializer_list<int64_t> s) const {
        return shape == std::vector<int64_t>(s);
    }
    std::string shape_str() const;
};

}  // namespace gpt2
