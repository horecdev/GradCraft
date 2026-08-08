#pragma once

#include "../tensor.hpp"
#include "shape_inference.hpp"

#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gradc {

    template <typename T>
    Tensor<T> lobotomized_broadcast_view(const Tensor<T>& source, const std::vector<int64_t>& target_shape) {
        const int64_t n_dim_orig = std::ssize(source.m_shape);
        const int64_t n_dim_target = std::ssize(target_shape);
        std::vector<int64_t> new_shape = std::vector<int64_t>(n_dim_target);
        std::vector<int64_t> new_strides = std::vector<int64_t>(n_dim_target);

        // align to the right
        for (int64_t i = 0; i < n_dim_orig; ++i) {
            new_shape[i + (n_dim_target - n_dim_orig)] = source.m_shape[i];
            new_strides[i + (n_dim_target - n_dim_orig)] = source.m_strides[i];
        }
        for (int64_t i = n_dim_target - n_dim_orig - 1; i >= 0; --i) { // fill leftovers (dont even have to check later)
            new_shape[i] = target_shape[i];
            new_strides[i] = 0;
        }

        for (int64_t i = (n_dim_target - n_dim_orig); i < n_dim_target; ++i) {
            if (target_shape[i] == new_shape[i]) {
                // literally leave everything as is
            }
            else if (new_shape[i] == 1) {
                new_shape[i] = target_shape[i];
                new_strides[i] = 0;
            }
            else {
                throw std::runtime_error("Violated broadcasting rules in lobotomized_broadcast_view().");
            }
        }
        return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false); // lobotomy (no past or future)
    }

    template <typename T>
    Tensor<T> lobotomized_transpose_view(const Tensor<T>& source, int64_t dim0, int64_t dim1) {
        const int64_t n_dim = std::ssize(source.m_shape);
        dim0 = normalize_axis(dim0, n_dim);
        dim1 = normalize_axis(dim1, n_dim);

        std::vector<int64_t> new_shape = source.m_shape;
        std::vector<int64_t> new_strides = source.m_strides;
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);

        Tensor<T> result = Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
        return result;
    }

    template <typename T>
    Tensor<T> lobotomized_reshape_view(const Tensor<T>& source, const std::vector<int64_t>& target_shape) {
        std::vector<int64_t> new_shape = std::vector<int64_t>(std::ssize(target_shape));
        std::vector<int64_t> new_strides = std::vector<int64_t>(std::ssize(target_shape));
        int64_t running_volume = 1;
        int64_t neg_one_idx = -1;

        int64_t total_volume = source.volume();
        
        for (int64_t i = 0; i < std::ssize(target_shape); ++i) {
            if (target_shape[i] == -1 && neg_one_idx == -1) {
                neg_one_idx = i;
            }
            else if (target_shape[i] == -1 && neg_one_idx != -1) {
                throw std::runtime_error("Cannot .reshape() with two or more unknown dimensions.");
            }
            else {
                new_shape[i] = target_shape[i];
                running_volume *= target_shape[i];
            }
        }

        int64_t unknown_dim;
        if (total_volume % running_volume != 0) {
            throw std::runtime_error("Invalid reshape parameters.");
        }
        else {
            unknown_dim = total_volume / running_volume;
        }

        if (neg_one_idx != -1) {
            new_shape[neg_one_idx] = unknown_dim;
        }

        new_strides[std::ssize(target_shape) - 1] = 1; // reshaping changes just where the rows break, not the order in which array is read. Therefore result is contiguous.
        for (int64_t i = std::ssize(target_shape) - 1; i > 0; --i) {
            new_strides[i - 1] = new_shape[i] * new_strides[i];
        }

        if (source.is_contiguous()) {
            return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
        }
        else {
            throw std::runtime_error("lobotomized_reshape_view invoked on a non-contiguous tensor.");
        }
    }

    template <typename T>
    Tensor<T> lobotomized_permute_view(const Tensor<T>& source, const std::vector<int64_t>& axes) {
        const int64_t n_dim = std::ssize(source.m_shape);
        if (std::ssize(axes) != n_dim) {
            throw std::runtime_error("permute() axes list size must match shape list size.");
        }
        std::vector<int64_t> normalized_axes = normalize_axes_vector(axes, n_dim);
        std::vector<int64_t> new_shape = std::vector<int64_t>(n_dim);
        std::vector<int64_t> new_strides = std::vector<int64_t>(n_dim);
        std::vector<bool> seen_axes = std::vector<bool>(n_dim, false);
        for (int64_t target_ax = 0; target_ax < n_dim; ++target_ax) {
            int64_t src_ax = normalized_axes[target_ax];
            if (seen_axes[src_ax]) {
                throw std::runtime_error("Passed at least one axis twice inside .permute()");
            }
            seen_axes[src_ax] = true;

            new_shape[target_ax] = source.m_shape[src_ax];
            new_strides[target_ax] = source.m_strides[src_ax];
        }

        return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
    }

    template <typename T> 
    Tensor<T> create_lobotomized_slice_view(const Tensor<T>& source, const std::vector<IndexDesc>& descriptors) {
        const int64_t n_dim = std::ssize(descriptors);
        if (n_dim != std::ssize(source.m_shape)) {
            throw std::out_of_range("Coordinate count does not match tensor dimensions.");
        }
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;
        int64_t new_offset = source.m_offset;
        new_shape.reserve(n_dim); // max amount of elements reserved upfront (evades dynamic reallocations)
        new_strides.reserve(n_dim);

        for (int64_t i = 0; i < n_dim; ++i) {
            if (descriptors[i].m_type == IndexType::All) {
                new_shape.push_back(source.m_shape[i]); // worked it out on paper
                new_strides.push_back(source.m_strides[i]);   
            }
            else if (descriptors[i].m_type == IndexType::Single) {
                int64_t coord = descriptors[i].m_single_val;
                coord = normalize_axis(coord, source.m_shape[i]);
                new_offset += coord * source.m_strides[i];
            }
            else if (descriptors[i].m_type == IndexType::Range) {
                int64_t step = descriptors[i].m_slice_val.step.value_or(1);
                if (step == 0) {
                    throw std::runtime_error("Slice cannot have a step of 0.");
                }
                // [start, stop) so start has can be 0 or shape[i] - 1 since its inclusive
                // stop can have -1 when its step < 0 or m_shape since its exclusive
                int64_t start = descriptors[i].m_slice_val.start.value_or(step > 0 ? 0 : source.m_shape[i] - 1);
                int64_t stop = descriptors[i].m_slice_val.stop.value_or(step > 0 ? source.m_shape[i] : -1);

                if (start < 0) { // start cannot be < 0 in any scenario
                    start += source.m_shape[i];
                }
                if (descriptors[i].m_slice_val.stop.has_value() && stop < 0) {
                    stop += source.m_shape[i];
                }

                if (start < 0 || start >= source.m_shape[i]) {
                    throw std::out_of_range("Range start is out of bounds.");
                }
                if (stop < -1 || stop > source.m_shape[i]) {
                    throw std::out_of_range("Range stop is out of bounds.");
                }
                if (step > 0 && start >= stop) {
                    throw std::runtime_error("Invalid forward slice: start >= stop with positive step.");
                }
                if (step < 0 && start <= stop) {
                    throw std::runtime_error("Invalid backward slice: start <= stop with negative step.");
                }

                int64_t n_dist;
                int64_t abs_step;
                if (step > 0) {
                    n_dist = stop - start;
                    abs_step = step;
                }
                else {
                    n_dist = start - stop;
                    abs_step = -step;
                }
                int64_t slice_len;
                if (n_dist % abs_step == 0) { // If step_size perfectly divides N, then we have N / step_size. Otherwise N / step_size + 1. (works for [a, b) range)
                    slice_len = n_dist / abs_step;
                }
                else {
                    slice_len = (n_dist / abs_step) + 1;
                }

                new_shape.push_back(slice_len);
                new_strides.push_back(source.m_strides[i] * step); // for neg strides just go backwards from the end. For pos just skip few elems.
                new_offset += start * source.m_strides[i]; // for neg starting from the back of the axis (say 4:2:-1). For pos from the front.
            }
        }

        Tensor<T> result = Tensor(std::move(new_shape), std::move(new_strides), new_offset, source.m_state->m_storage, false);
        
        return result;
    }

    template <typename T>
    Tensor<T> lobotomized_squeeze_view(const Tensor<T>& source, std::optional<int64_t> target_dim = std::nullopt) {
        const int64_t n_dim = std::ssize(source.m_shape);
        if (n_dim == 0) {
            throw std::runtime_error("Tried squeezing a scalar (0D Tensor)");
        }
        
        std::vector<int64_t> new_shape;
        new_shape.reserve(n_dim);
        std::vector<int64_t> new_strides;
        new_strides.reserve(n_dim);

        if (target_dim.has_value()) {
            target_dim.value() = normalize_axis(target_dim.value(), n_dim);
            if (source.m_shape[target_dim.value()] != 1) {
                throw std::runtime_error("Tried squeezing a dimension whose size isnt 1.");
            }
            for (int64_t i = 0; i < n_dim; ++i) {
                if (i != target_dim) {
                    new_shape.push_back(source.m_shape[i]);
                    new_strides.push_back(source.m_strides[i]);
                }
            }

            return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
        }
        else {
            for (int64_t i = 0; i < n_dim; ++i) {
                if (source.m_shape[i] != 1) {
                    new_shape.push_back(source.m_shape[i]);
                    new_strides.push_back(source.m_strides[i]);
                }
            }

            return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
        }
    }

    template <typename T>
    Tensor<T> lobotomized_unsqueeze_view(const Tensor<T>& source, int64_t dim = 0) {
        const int64_t n_dim = std::ssize(source.m_shape);
        dim = normalize_axis(dim, n_dim + 1);
        
        std::vector<int64_t> new_shape;
        new_shape.reserve(n_dim + 1);
        std::vector<int64_t> new_strides;
        new_strides.reserve(n_dim + 1);

        if (dim == n_dim) {
            for (int64_t i = 0; i < n_dim; ++i) {
                new_shape.push_back(source.m_shape[i]);
                new_strides.push_back(source.m_strides[i]);
            }

            new_shape.push_back(1);
            new_strides.push_back(1);

            return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
        }

        else {
            for (int64_t i = 0; i < n_dim; ++i) {
                if (i == dim) {
                    new_shape.push_back(1);
                    new_strides.push_back(source.m_shape[i] * source.m_strides[i]);
                }
                
                new_shape.push_back(source.m_shape[i]);
                new_strides.push_back(source.m_strides[i]);
            }

            return Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
        }
    }


}