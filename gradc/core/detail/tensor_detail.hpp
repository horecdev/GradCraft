#pragma once

#include "../../backend/dispatcher.hpp"
#include "../tensor.hpp"
#include "shape_inference.hpp"

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gradc {

    template <typename T>
    Tensor<T> unbroadcast_grad(const Tensor<T>& raw_grad, const std::vector<int64_t>& orig_shape) {
        std::vector<int64_t> broadcast_axes = find_broadcast_axes(raw_grad.m_shape, orig_shape);

        if (broadcast_axes.empty()) {
            return raw_grad;
        }

        ReductionMetadata red_meta = infer_reduction_metadata(raw_grad.m_shape, broadcast_axes, false);
        Tensor<T> reduced = Tensor<T>(orig_shape, raw_grad.device(), uninitialized);
        dispatch(raw_grad.device(), ReduceOp::Sum, red_meta, reduced, raw_grad);
        return reduced; 

        // contiguous tensor (strides must be generated and shape must be kept as the one of parent)
        // We keepdims=false so [2, 1] broadcast into [2, 5] turns into [2]. To fix that, we just use the same shape as the original (parent).
        // It works because reduction operation returns a contiguous tensor. You can just fake dimensions of one, and since we dont know which one we reduced
        // and which one were here beforehand, just collapse both and artificially add them later.
        // From now on since shape/strides of reduced isnt used in reduction operation, its done at the start.
    }

    template <typename T, typename U>
    auto promote_to_common(Tensor<T> left, Tensor<U> right) {
        // .template is a promise to the compiler that cast is a template (it doesnt know what T is during first read, 
        // and since its dependent - inside Tensor class, not on its own - it assumes its a variable by default. .template forces it to look at it as a template.)
        using PromotedT = std::common_type_t<T, U>;

        Tensor<PromotedT> p_left;
        Tensor<PromotedT> p_right;

        if constexpr (std::is_same_v<T, PromotedT>) {
            p_left = std::move(left);
        }
        else {
            p_left = left.template cast<PromotedT>();
        }

        if constexpr (std::is_same_v<U, PromotedT>) {
            p_right = std::move(right);
        }
        else {
            p_right = right.template cast<PromotedT>();
        }

        return std::make_pair(std::move(p_left), std::move(p_right));
    }

    inline std::pair<Device, cudaMemcpyKind> infer_cuda_memcpy_device_kind(Device src, Device dst) {
        if (src.is_cpu() && dst.is_cuda()) {
            return std::make_pair(dst, cudaMemcpyHostToDevice);
        }
        else if (src.is_cuda() && dst.is_cpu()) {
            return std::make_pair(src, cudaMemcpyDeviceToHost);
        }
        else if (src.is_cuda() && dst.is_cuda()) {
            return std::make_pair(dst, cudaMemcpyDeviceToDevice);
        }
        else {
            throw std::runtime_error("Unknown transfer direction over the PCIe bus");
        }
    }

    inline void throw_cuda_memcpy_error(cudaMemcpyKind kind) {
        switch (kind) {
            case cudaMemcpyHostToDevice: {throw std::runtime_error("Copying data from Host to Device failed.");}
            case cudaMemcpyDeviceToHost: {throw std::runtime_error("Copying data from Device to Host failed.");}
            case cudaMemcpyDeviceToDevice: {throw std::runtime_error("Copying data from Device to Device failed.");}
            case cudaMemcpyHostToHost: {throw std::runtime_error("Copying data from Host to Host failed.");}
            case cudaMemcpyDefault: {throw std::runtime_error("Default cudamemcpy failed.");}
        }
    }

    template <typename T>
    inline BLASGEMMMeta infer_blas_meta(Tensor<T> left, Tensor<T> right, bool accumulate) { // takes left, right by value. 
        if (std::ssize(right.shape()) != 2) {
            throw std::runtime_error("B in A @ B must be 2 dimensional.");
        }
        if (left.shape().back() != right.shape().front()) {
            throw std::runtime_error("Wrong dimensions for matmul: left.back() != right.front()");
        }
        
        int64_t n_dim = std::ssize(left.shape());
        BLASGEMMMeta blas_meta;
        if (accumulate) {
            blas_meta.alpha = 1.0;
            blas_meta.beta = 1.0;
        }
        else {
            blas_meta.alpha = 1.0;
            blas_meta.beta = 0.0;
        }

        std::vector<int64_t> left_shape_except_rightmost = std::vector<int64_t>(left.shape().begin(), left.shape().end() - 1);
        std::vector<int64_t> left_strides_except_rightmost = std::vector<int64_t>(left.strides().begin(), left.strides().end() - 1);
        FusedView initial_fuse_result = fuse_dimensions(left_shape_except_rightmost, {left_strides_except_rightmost});

        Tensor<T> left_locally_contig;
        if (std::ssize(initial_fuse_result.shared_shape) != 1) {
            left_locally_contig = left.contiguous();
            std::vector<int64_t> left_locally_contig_shape_except_rightmost(left_locally_contig.shape().begin(), left_locally_contig.shape().end() - 1);
            std::vector<int64_t> left_locally_contig_strides_except_rightmost(left_locally_contig.strides().begin(), left_locally_contig.strides().end() - 1);
            FusedView locally_contig_fuse = fuse_dimensions(left_locally_contig_shape_except_rightmost, {left_locally_contig_strides_except_rightmost});
            left_locally_contig.m_shape = locally_contig_fuse.shared_shape;
            left_locally_contig.m_strides = locally_contig_fuse.strides[0];
            left_locally_contig.m_shape.push_back(left.shape().back());
            left_locally_contig.m_strides.push_back(left.strides().back());
        }
        else {
            left_locally_contig = left;
            left_locally_contig.m_shape = initial_fuse_result.shared_shape;
            left_locally_contig.m_strides = initial_fuse_result.strides[0];
            left_locally_contig.m_shape.push_back(left.shape().back());
            left_locally_contig.m_strides.push_back(left.strides().back());
        }
        // after this step we know for SURE that left is 2 dimensional (B * T, C) and right is (C, H)

        // Now it has to be either transposed or not transposed. If none of the dimensions are 1, force contiguity (on both)

        std::vector<int64_t> result_shape = left.shape();
        result_shape[n_dim - 1] = right.shape()[1];

        blas_meta.result_shape = result_shape;
        blas_meta.M = left_locally_contig.shape()[0];
        blas_meta.K = right.shape()[0];
        blas_meta.N = right.shape()[1];

        if (left_locally_contig.strides()[1] == 1 && left_locally_contig.strides()[0] > 0) { // blas does not accept negative leading dims
            blas_meta.lda = left_locally_contig.strides()[0];
            blas_meta.left_is_transposed = MatrixTensorOp::Normal;
        }
        else if (left_locally_contig.strides()[0] == 1 && left_locally_contig.strides()[1] > 0) { // transposed
            blas_meta.lda = left.strides()[1];
            blas_meta.left_is_transposed = MatrixTensorOp::Transposed;
        }
        else {
            left_locally_contig = left_locally_contig.contiguous();
            blas_meta.lda = left_locally_contig.strides()[0];
            blas_meta.left_is_transposed = MatrixTensorOp::Normal;
        }
        // left_locally_contig has whole graph of left

        Tensor<T> safe_right = right; // has whole story of right
        if (right.strides()[1] == 1 && right.strides()[0] > 0) { // blas does not accept negative leading dims
            blas_meta.ldb = right.strides()[0];
            blas_meta.right_is_transposed = MatrixTensorOp::Normal;
        }
        else if (right.strides()[0] == 1 && right.strides()[1] > 0) { // transposed
            blas_meta.ldb = left.strides()[1];
            blas_meta.right_is_transposed = MatrixTensorOp::Transposed;
        }
        else {
            safe_right = right.contiguous();
            blas_meta.ldb = right.strides()[0];
            blas_meta.right_is_transposed = MatrixTensorOp::Normal;
        }

        blas_meta.ldc = blas_meta.N;

        return std::make_pair(std::make_pair(std::move(left_locally_contig), std::move(safe_right)), blas_meta);
    }
}