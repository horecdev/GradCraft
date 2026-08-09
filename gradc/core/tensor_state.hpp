#pragma once

#include "../graph/node.hpp"
#include "storage.hpp"

#include <memory>
#include <optional>
#include <utility>
#include <vector>
// doesnt need tensor.hpp cuz tensor_state.hpp is only ever included at its bottom

namespace gradc {

    template <typename T> class Node;

    struct TensorStateBase {
        public:
            virtual std::vector<TensorStateBase*> get_dependencies() const = 0;

            virtual void backward(bool retain_graph) const = 0;

            virtual void clear_grad_if_non_leaf() = 0;
        };

    template <typename T>
    struct TensorState : public TensorStateBase {
        std::shared_ptr<Storage<T>> m_storage; // nodes can share just storage
        std::unique_ptr<Node<T>> m_creation_op; // it makes no sense to share m_op, but not storage.
        bool m_is_realized;
        std::optional<Tensor<T>> m_grad = std::nullopt; // every TensorState has grad. Even reshapes.


        TensorState() : m_storage(std::make_shared<Storage<T>>()), m_creation_op(nullptr), m_is_realized(false) {}
        TensorState(int64_t size, T init_val = T(0), Device device = Device(DeviceType::CPU), bool allocate = true, bool fill = true) : m_storage(std::make_shared<Storage<T>>(size, init_val, device, allocate, fill)), m_creation_op(nullptr), m_is_realized(allocate) {}
        TensorState(std::shared_ptr<Storage<T>> storage) : m_storage(std::move(storage)), m_creation_op(nullptr), m_is_realized(false) {} // copy tensor storage, set m_r_op to be nothing
        TensorState(std::shared_ptr<Storage<T>> storage, std::unique_ptr<Node<T>> realize_op) : m_storage(std::move(storage)), m_creation_op(std::move(realize_op)), m_is_realized(false) {}


        std::vector<TensorStateBase*> get_dependencies() const override {
            if (m_creation_op == nullptr) {
                return std::vector<TensorStateBase*>();
            }
            return m_creation_op->get_input_states();
        }

        void backward(bool retain_graph) const override {
            if (m_creation_op != nullptr) {
                m_creation_op->backward(m_grad.value(), retain_graph);
            }
        }

        void clear_grad_if_non_leaf() override {
            if (m_creation_op != nullptr) {
                m_grad = std::nullopt;
            }
        }
    };
}