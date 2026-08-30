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

        RedMeta red_meta = infer_red_meta(raw_grad.m_shape, broadcast_axes, false);
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
    inline auto infer_blas_normal_meta(Tensor<T> left, Tensor<T> right, bool accumulate) { // takes left, right by value. 
        if (std::ssize(right.shape()) != 2) {
            throw std::runtime_error("B in A @ B must be 2 dimensional.");
        }
        if (left.shape().back() != right.shape().front()) {
            throw std::runtime_error("Wrong dimensions for matmul: left.back() != right.front()");
        }
        
        int64_t n_dim = std::ssize(left.shape());
        NMMMeta blas_meta;
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
        FusedView initial_fuse_result = fuse_dimensions(left_shape_except_rightmost, {&left_strides_except_rightmost});

        Tensor<T> left_locally_contig;
        if (std::ssize(initial_fuse_result.shared_shape) != 1) {
            // path where you cannot fuse into 2D immediately
            left_locally_contig = left.contiguous();
            std::vector<int64_t> left_locally_contig_shape_except_rightmost(left_locally_contig.shape().begin(), left_locally_contig.shape().end() - 1);
            std::vector<int64_t> left_locally_contig_strides_except_rightmost(left_locally_contig.strides().begin(), left_locally_contig.strides().end() - 1);
            FusedView locally_contig_fuse = fuse_dimensions(left_locally_contig_shape_except_rightmost, {&left_locally_contig_strides_except_rightmost});
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
            blas_meta.left_op = MatrixTensorOp::Normal;

            if (blas_meta.M == 1 && blas_meta.lda < blas_meta.K) { // purely for satisfying BLAS which says: lda > K. If not, fails.
                // if shape == 1 then this lda will never even be used.
                blas_meta.lda = blas_meta.K;
            }
        }
        else if (left_locally_contig.strides()[0] == 1 && left_locally_contig.strides()[1] > 0) { // transposed
            blas_meta.lda = left_locally_contig.strides()[1];
            blas_meta.left_op = MatrixTensorOp::Transposed;

            if (blas_meta.K == 1 && blas_meta.lda < blas_meta.M) { // same but now its [K, M] instead of [M, K]
                blas_meta.lda = blas_meta.M;
            }
        }
        else {
            // you CAN flatten into 2D (say it comes in as 2D) but it doesnt have lda=1 or ldb=1
            left_locally_contig = left_locally_contig.contiguous();
            blas_meta.lda = left_locally_contig.strides()[0];
            blas_meta.left_op = MatrixTensorOp::Normal;
        }
        // left_locally_contig has whole graph of left

        Tensor<T> safe_right = right; // has whole story of right
        if (right.strides()[1] == 1 && right.strides()[0] > 0) { // blas does not accept negative leading dims
            blas_meta.ldb = right.strides()[0];
            blas_meta.right_op = MatrixTensorOp::Normal;

            if (blas_meta.K == 1 && blas_meta.ldb < blas_meta.N) {
                blas_meta.ldb = blas_meta.N;
            }
        }
        else if (right.strides()[0] == 1 && right.strides()[1] > 0) { // transposed
            blas_meta.ldb = right.strides()[1];
            blas_meta.right_op = MatrixTensorOp::Transposed;

            if (blas_meta.N == 1 && blas_meta.ldb < blas_meta.K) {
                blas_meta.ldb = blas_meta.K;
            }
        }
        else {
            safe_right = right.contiguous();
            blas_meta.ldb = right.strides()[0];
            blas_meta.right_op = MatrixTensorOp::Normal;
        }

        blas_meta.ldc = blas_meta.N;

        return std::make_pair(std::make_pair(std::move(left_locally_contig), std::move(safe_right)), blas_meta);
    }

    template <typename T>
    inline auto infer_blas_batched_meta(Tensor<T> left, Tensor<T> right, bool accumulate) {
        if (std::ssize(left.shape()) <= 2 || std::ssize(right.shape()) <= 2) {
            throw std::runtime_error("left and right shapes in infer_blas_batched_meta must be >= 3D");
        }

        int64_t left_n_dim = std::ssize(left.shape());
        int64_t right_n_dim = std::ssize(right.shape());
        BMMMeta blas_meta;

        if (accumulate) {
            blas_meta.alpha = 1.0;
            blas_meta.beta = 1.0;
        }
        else {
            blas_meta.alpha = 1.0;
            blas_meta.beta = 0.0;
        }

        std::vector<int64_t> left_batch_shape = std::vector<int64_t>(left.shape().begin(), left.shape().end() - 2);
        std::vector<int64_t> left_batch_strides = std::vector<int64_t>(left.strides().begin(), left.strides().end() - 2);
        FusedView init_left_fuse = fuse_dimensions(left_batch_shape, {&left_batch_strides});

        Tensor<T> left_locally_contig;
        if (std::ssize(init_left_fuse.shared_shape) != 1) { // unable to fuse into single batch dim
            left_locally_contig = left.contiguous();
            std::vector<int64_t> left_locally_contig_batch_shape(left_locally_contig.shape().begin(), left_locally_contig.shape().end() - 2);
            std::vector<int64_t> left_locally_contig_batch_strides(left_locally_contig.strides().begin(), left_locally_contig.strides().end() - 2);
            FusedView left_locally_contig_fuse = fuse_dimensions(left_locally_contig_batch_shape, {&left_locally_contig_batch_strides});

            left_locally_contig.m_shape = left_locally_contig_fuse.shared_shape; // now its just B
            left_locally_contig.m_strides = left_locally_contig_fuse.strides[0]; // also just B
            left_locally_contig.m_shape.push_back(left.shape()[left_n_dim - 2]);
            left_locally_contig.m_shape.push_back(left.shape()[left_n_dim - 1]); // now its B, T, C
            left_locally_contig.m_strides.push_back(left.strides()[left_n_dim - 2]);
            left_locally_contig.m_strides.push_back(left.strides()[left_n_dim - 1]);
        }
        else {
            left_locally_contig = left;
            left_locally_contig.m_shape = init_left_fuse.shared_shape;
            left_locally_contig.m_strides = init_left_fuse.strides[0];
            left_locally_contig.m_shape.push_back(left.shape()[left_n_dim - 2]);
            left_locally_contig.m_shape.push_back(left.shape()[left_n_dim - 1]);
            left_locally_contig.m_strides.push_back(left.strides()[left_n_dim - 2]);
            left_locally_contig.m_strides.push_back(left.strides()[left_n_dim - 1]);
        }
        // left now left is 3D (B, T, C)

        // repeat for right:

        std::vector<int64_t> right_batch_shape = std::vector<int64_t>(right.shape().begin(), right.shape().end() - 2);
        std::vector<int64_t> right_batch_strides = std::vector<int64_t>(right.strides().begin(), right.strides().end() - 2);
        FusedView init_right_fuse = fuse_dimensions(right_batch_shape, {&right_batch_strides});

        Tensor<T> right_locally_contig;
        if (std::ssize(init_right_fuse.shared_shape) != 1) { 
            right_locally_contig = right.contiguous();
            std::vector<int64_t> right_locally_contig_batch_shape(right_locally_contig.shape().begin(), right_locally_contig.shape().end() - 2);
            std::vector<int64_t> right_locally_contig_batch_strides(right_locally_contig.strides().begin(), right_locally_contig.strides().end() - 2);
            FusedView right_locally_contig_fuse = fuse_dimensions(right_locally_contig_batch_shape, {&right_locally_contig_batch_strides});

            right_locally_contig.m_shape = right_locally_contig_fuse.shared_shape; 
            right_locally_contig.m_strides = right_locally_contig_fuse.strides[0];
            right_locally_contig.m_shape.push_back(right.shape()[right_n_dim - 2]);
            right_locally_contig.m_shape.push_back(right.shape()[right_n_dim - 1]);
            right_locally_contig.m_strides.push_back(right.strides()[right_n_dim - 2]);
            right_locally_contig.m_strides.push_back(right.strides()[right_n_dim - 1]);
        }
        else {
            right_locally_contig = right;
            right_locally_contig.m_shape = init_right_fuse.shared_shape;
            right_locally_contig.m_strides = init_right_fuse.strides[0];
            right_locally_contig.m_shape.push_back(right.shape()[right_n_dim - 2]);
            right_locally_contig.m_shape.push_back(right.shape()[right_n_dim - 1]);
            right_locally_contig.m_strides.push_back(right.strides()[right_n_dim - 2]);
            right_locally_contig.m_strides.push_back(right.strides()[right_n_dim - 1]);
        }

        // now both tensors are B, T, C. Both have ENTIRE story of left and right

        // START OFF FROM HERE (MORE CONTIGUITY FORCING IF NONE OF THE T, C STRIDES ARE 1, INFERING LDA, TRANSPOSALS)

        if (left_locally_contig.shape()[0] != right_locally_contig.shape()[0]) {
            throw std::runtime_error("Batch size mistmatch during BMM");
        }

        blas_meta.batch_count = left_locally_contig.shape()[0];
        
        std::vector<int64_t> result_shape = left.shape();
        result_shape.back() = right.shape().back();

        blas_meta.result_shape = result_shape;
        blas_meta.M = left_locally_contig.shape()[1];
        blas_meta.K = right_locally_contig.shape()[1];
        blas_meta.N = right_locally_contig.shape()[2];

        if (left_locally_contig.strides()[2] == 1 && left_locally_contig.strides()[1] > 0 && left_locally_contig.strides()[0] > 0) {
            blas_meta.lda = left_locally_contig.strides()[1];
            if (blas_meta.M == 1 && blas_meta.lda < blas_meta.K) {
                blas_meta.lda = blas_meta.K;
            }
            blas_meta.left_op = MatrixTensorOp::Normal;
            blas_meta.stride_a = left_locally_contig.strides()[0];
        }
        else if (left_locally_contig.strides()[1] == 1 && left_locally_contig.strides()[2] > 0 && left_locally_contig.strides()[0] > 0) {
            blas_meta.lda = left_locally_contig.strides()[2];
            if (blas_meta.K == 1 && blas_meta.lda < blas_meta.M) {
                blas_meta.lda = blas_meta.M;
            }
            blas_meta.left_op = MatrixTensorOp::Transposed;
            blas_meta.stride_a = left_locally_contig.strides()[0];
        }
        else {
            left_locally_contig = left_locally_contig.contiguous();
            blas_meta.lda = left_locally_contig.strides()[1];
            blas_meta.left_op = MatrixTensorOp::Normal;
            blas_meta.stride_a = left_locally_contig.strides()[0];
        }
        // left_locally_contig now is 100% viable for matmul 

        if (right_locally_contig.strides()[2] == 1 && right_locally_contig.strides()[1] > 0 && right_locally_contig.strides()[0] > 0) {
            blas_meta.ldb = right_locally_contig.strides()[1];
            if (blas_meta.K == 1 && blas_meta.ldb < blas_meta.N) {
                blas_meta.ldb = blas_meta.N;
            }
            blas_meta.right_op = MatrixTensorOp::Normal;
            blas_meta.stride_b = right_locally_contig.strides()[0];
        }
        else if (right_locally_contig.strides()[1] == 1 && right_locally_contig.strides()[2] > 0 && right_locally_contig.strides()[0] > 0) {
            blas_meta.ldb = right_locally_contig.strides()[2];
            if (blas_meta.N == 1 & blas_meta.ldb < blas_meta.K) {
                blas_meta.ldb = blas_meta.K;
            }
            blas_meta.right_op = MatrixTensorOp::Transposed;
            blas_meta.stride_b = right_locally_contig.strides()[0];
        }
        else {
            right_locally_contig = right_locally_contig.contiguous();
            blas_meta.ldb = right_locally_contig.strides()[1];
            blas_meta.right_op = MatrixTensorOp::Normal;
            blas_meta.stride_b = right_locally_contig.strides()[0];
        }
        // right_locally_contig now is 100% viable for matmul 

        blas_meta.ldc = blas_meta.N;
        blas_meta.stride_c = blas_meta.M * blas_meta.N;

        return std::make_pair(std::make_pair(std::move(left_locally_contig), std::move(right_locally_contig)), blas_meta);
    }
}