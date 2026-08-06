#pragma once

#include "../core/detail/shape_inference.hpp"
#include "../core/detail/tensor_detail.hpp"
#include "../core/tensor.hpp"
#include "../graph/nodes/math_nodes.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gradc {
    #pragma region OUT OF PLACE

    template <typename T, typename U>
    auto operator+(Tensor<T> left, Tensor<U> right) {
        Device target_device = infer_assert_device(left, right);
        
        using PromotedT = std::common_type_t<T, U>;

        auto [p_left, p_right] = promote_to_common(std::move(left), std::move(right));

        std::vector<int64_t> target_shape;
        if (p_left.m_shape != p_right.m_shape) {
            target_shape = infer_broadcast(p_left.m_shape, p_right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = p_left.m_shape;
        }

        bool requires_grad = p_left.m_requires_grad || p_right.m_requires_grad;
        Tensor<PromotedT> new_tensor = Tensor<PromotedT>(target_shape, requires_grad, lazy, target_device);
        new_tensor.m_state->m_creation_op = std::make_unique<AddNode<PromotedT>>(std::move(p_left), std::move(p_right), std::move(target_shape));
        return new_tensor;
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    auto operator+(Tensor<T> left, U right_val) { // Tensor + scalar
        return std::move(left) + Tensor<U>(right_val, left.device());
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<T>
    auto operator+(T left_val, Tensor<U> right) { // scalar + Tensor
        return Tensor<T>(left_val, right.device()) + std::move(right);
    }

    template <typename T, typename U>
    auto operator-(Tensor<T> left, Tensor<U> right) {
        Device target_device = infer_assert_device(left, right);
        
        using PromotedT = std::common_type_t<T, U>;

        auto [p_left, p_right] = promote_to_common(std::move(left), std::move(right));

        std::vector<int64_t> target_shape;
        if (p_left.m_shape != p_right.m_shape) {
            target_shape = infer_broadcast(p_left.m_shape, p_right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = p_left.m_shape;
        }

        bool requires_grad = p_left.m_requires_grad || p_right.m_requires_grad;
        Tensor<PromotedT> new_tensor = Tensor<PromotedT>(target_shape, requires_grad, lazy, target_device);
        new_tensor.m_state->m_creation_op = std::make_unique<SubNode<PromotedT>>(std::move(p_left), std::move(p_right), std::move(target_shape));
        return new_tensor;
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    auto operator-(Tensor<T> left, U right_val) { // Tensor + scalar
        return std::move(left) - Tensor<U>(right_val, left.device());
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<T>
    auto operator-(T left_val, Tensor<U> right) { // scalar + Tensor
        return Tensor<T>(left_val, right.device()) - std::move(right);
    }

    template <typename T, typename U>
    auto operator*(Tensor<T> left, Tensor<U> right) {
        Device target_device = infer_assert_device(left, right);
        
        using PromotedT = std::common_type_t<T, U>;

        auto [p_left, p_right] = promote_to_common(std::move(left), std::move(right));

        std::vector<int64_t> target_shape;
        if (p_left.m_shape != p_right.m_shape) {
            target_shape = infer_broadcast(p_left.m_shape, p_right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = p_left.m_shape;
        }

        bool requires_grad = p_left.m_requires_grad || p_right.m_requires_grad;
        Tensor<PromotedT> new_tensor = Tensor<PromotedT>(target_shape, requires_grad, lazy, target_device);
        new_tensor.m_state->m_creation_op = std::make_unique<MulNode<PromotedT>>(std::move(p_left), std::move(p_right), std::move(target_shape));
        return new_tensor;
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    auto operator*(Tensor<T> left, U right_val) { // Tensor + scalar
        return std::move(left) * Tensor<U>(right_val, left.device());
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<T>
    auto operator*(T left_val, Tensor<U> right) { // scalar + Tensor
        return Tensor<T>(left_val, right.device()) * std::move(right);
    }

    template <typename T, typename U>
    auto operator/(Tensor<T> left, Tensor<U> right) {
        Device target_device = infer_assert_device(left, right);
        
        using PromotedT = std::common_type_t<T, U>;

        auto [p_left, p_right] = promote_to_common(std::move(left), std::move(right));

        std::vector<int64_t> target_shape;
        if (p_left.m_shape != p_right.m_shape) {
            target_shape = infer_broadcast(p_left.m_shape, p_right.m_shape); // crashes if incompatible on the graph-building stage.
        }
        else {
            target_shape = p_left.m_shape;
        }

        bool requires_grad = p_left.m_requires_grad || p_right.m_requires_grad;
        Tensor<PromotedT> new_tensor = Tensor<PromotedT>(target_shape, requires_grad, lazy, target_device);
        new_tensor.m_state->m_creation_op = std::make_unique<DivNode<PromotedT>>(std::move(p_left), std::move(p_right), std::move(target_shape));
        return new_tensor;
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    auto operator/(Tensor<T> left, U right_val) { // Tensor + scalar
        return std::move(left) / Tensor<U>(right_val, left.device());
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<T>
    auto operator/(T left_val, Tensor<U> right) { // scalar + Tensor
        return Tensor<T>(left_val, right.device()) / std::move(right);
    }

    #pragma endregion OUT OF PLACE

    #pragma region IN PLACE

    template <typename T, typename U>
    Tensor<T>& operator+=(Tensor<T>& main, Tensor<U> other) {
        Device target_device = infer_assert_device(main, other);

        using PromotedT = std::common_type_t<T, U>;
        Tensor<PromotedT> p_other;

        static_assert(std::is_same_v<T, PromotedT>, "FATAL: Cannot promote type of main tensor during in-place operation.");

        if constexpr (!std::is_same_v<U, PromotedT>) {
            p_other = other.template cast<PromotedT>();
        }
        else {
            p_other = std::move(other);
        }

        if (main.m_requires_grad && main.m_state->m_creation_op == nullptr) {
            throw std::runtime_error("Cannot mutate leaf tensor that requires gradients in-place.");
        }

        if (main.m_shape != p_other.m_shape) { // Validates during graph building
            if (!can_broadcast(other.m_shape, main.m_shape)) {
                throw std::runtime_error("Could not broadcast RHS to match LHS during in-place operation.");
            }
        }

        bool requires_grad = main.m_requires_grad || p_other.m_requires_grad;
        std::shared_ptr<TensorState<T>> new_state = std::make_shared<TensorState<T>>(main._get_storage().size(), T(0), target_device, false);
        new_state->m_creation_op = std::make_unique<AddNode<T>>(main, p_other, main.m_shape);
        main.m_state = std::move(new_state);
        main.m_requires_grad = requires_grad;
        return main;
        
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    Tensor<T>& operator+=(Tensor<T>& main, U other_val) {
        return main += Tensor<U>(other_val, main.device());
    }

    template <typename T, typename U>
    Tensor<T>& operator-=(Tensor<T>& main, Tensor<U> other) {
        Device target_device = infer_assert_device(main, other);

        using PromotedT = std::common_type_t<T, U>;
        Tensor<PromotedT> p_other;

        static_assert(std::is_same_v<T, PromotedT>, "FATAL: Cannot promote type of main tensor during in-place operation.");

        if constexpr (!std::is_same_v<U, PromotedT>) {
            p_other = other.template cast<PromotedT>();
        }
        else {
            p_other = std::move(other);
        }

        if (main.m_requires_grad && main.m_state->m_creation_op == nullptr) {
            throw std::runtime_error("Cannot mutate leaf tensor that requires gradients in-place.");
        }

        if (main.m_shape != p_other.m_shape) { // Validates during graph building
            if (!can_broadcast(other.m_shape, main.m_shape)) {
                throw std::runtime_error("Could not broadcast RHS to match LHS during in-place operation.");
            }
        }

        bool requires_grad = main.m_requires_grad || p_other.m_requires_grad;
        std::shared_ptr<TensorState<T>> new_state = std::make_shared<TensorState<T>>(main._get_storage().size(), T(0), target_device, false);
        new_state->m_creation_op = std::make_unique<SubNode<T>>(main, p_other, main.m_shape);
        main.m_state = std::move(new_state);
        main.m_requires_grad = requires_grad;
        return main;
        
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    Tensor<T>& operator-=(Tensor<T>& main, U other_val) {
        return main -= Tensor<U>(other_val, main.device());
    }

    template <typename T, typename U>
    Tensor<T>& operator*=(Tensor<T>& main, const Tensor<U> other) {
        Device target_device = infer_assert_device(main, other);

        using PromotedT = std::common_type_t<T, U>;
        Tensor<PromotedT> p_other;

        static_assert(std::is_same_v<T, PromotedT>, "FATAL: Cannot promote type of main tensor during in-place operation.");
        
        if constexpr (!std::is_same_v<U, PromotedT>) {
            p_other = other.template cast<PromotedT>();
        }
        else {
            p_other = std::move(other);
        }

        if (main.m_requires_grad && main.m_state->m_creation_op == nullptr) { // we are using toposort to get rid of temporary results using leaf nodes as a stop-point, and in-place math literally moves it inside the graph.
            throw std::runtime_error("Cannot mutate leaf tensor that requires gradients in-place.");
        }

        if (main.m_shape != other.m_shape) { // Validates on the graph building
            if (!can_broadcast(other.m_shape, main.m_shape)) {
                throw std::runtime_error("Could not broadcast RHS to match LHS during in-place operation.");
            }
        }

        bool requires_grad = main.m_requires_grad || p_other.m_requires_grad;
        std::shared_ptr<TensorState<T>> new_state = std::make_shared<TensorState<T>>(main._get_storage().size(), T(0), target_device, false);
        new_state->m_creation_op = std::make_unique<MulNode<T>>(main, p_other, main.m_shape);
        main.m_state = std::move(new_state);
        main.m_requires_grad = requires_grad;
        return main;
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    Tensor<T>& operator*=(Tensor<T>& main, U other_val) {
        return main *= Tensor<U>(other_val, main.device());
    }

    template <typename T, typename U>
    Tensor<T>& operator/=(Tensor<T>& main, const Tensor<U> other) {
        Device target_device = infer_assert_device(main, other);

        using PromotedT = std::common_type_t<T, U>;
        Tensor<PromotedT> p_other;

        static_assert(std::is_same_v<T, PromotedT>, "FATAL: Cannot promote type of main tensor during in-place operation.");
        
        if constexpr (!std::is_same_v<U, PromotedT>) {
            p_other = other.template cast<PromotedT>();
        }
        else {
            p_other = std::move(other);
        }

        if (main.m_requires_grad && main.m_state->m_creation_op == nullptr) { // we are using toposort to get rid of temporary results using leaf nodes as a stop-point, and in-place math literally moves it inside the graph.
            throw std::runtime_error("Cannot mutate leaf tensor that requires gradients in-place.");
        }

        if (main.m_shape != other.m_shape) { // Validates on the graph building
            if (!can_broadcast(other.m_shape, main.m_shape)) {
                throw std::runtime_error("Could not broadcast RHS to match LHS during in-place operation.");
            }
        }

        bool requires_grad = main.m_requires_grad || p_other.m_requires_grad;
        std::shared_ptr<TensorState<T>> new_state = std::make_shared<TensorState<T>>(main._get_storage().size(), T(0), target_device, false);
        new_state->m_creation_op = std::make_unique<DivNode<T>>(main, p_other, main.m_shape);
        main.m_state = std::move(new_state);
        main.m_requires_grad = requires_grad;
        return main;
    }

    template <typename T, typename U>
    requires std::is_arithmetic_v<U>
    Tensor<T>& operator/=(Tensor<T>& main, U other_val) {
        return main *= Tensor<U>(other_val, main.device());
    }

    #pragma endregion IN PLACE

    #pragma region OTHER

    template <typename T>
    Tensor<T> Tensor<T>::operator-() const {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<NegNode<T>>(*this);
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::exp() const requires std::is_floating_point_v<T> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<ExpNode<T>>(*this);
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::log() const requires std::is_floating_point_v<T> {
        Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device());
        result.m_state->m_creation_op = std::make_unique<LogNode<T>>(*this);
        return result;
    }

    template <typename T, typename U>
    requires (std::is_floating_point_v<T> && std::is_floating_point_v<U>)
    inline Tensor<T> matmul(Tensor<T> left, Tensor<U> right) {
        Device target_device = infer_assert_device(left, right);
        
        using PromotedT = std::common_type_t<T, U>;
        auto [p_left, p_right] = promote_to_common(std::move(left), std::move(right));

        bool requires_grad = p_left.requires_grad() || p_left.requires_grad();

        std::pair<std::pair<Tensor<T>, Tensor<T>>, BLASGEMMMeta> gemm_prep = infer_blas_meta(std::move(p_left), std::move(p_right), false);
        BLASGEMMMeta blas_meta = gemm_prep.second;
        Tensor<T> safe_left = std::move(gemm_prep.first.first);
        Tensor<T> safe_right = std::move(gemm_prep.first.second);

        Tensor<PromotedT> result = Tensor<PromotedT>(blas_meta.result_shape, requires_grad, lazy, target_device);
        result.m_state->m_creation_op = std::make_unique<MatMulNode<PromotedT>>(std::move(safe_left), std::move(safe_right), blas_meta);
        return result;
    }


    #pragma endregion OTHER
}