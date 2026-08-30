#pragma once

#include "gradc/nn/base/parameter.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class Optimizer {
        protected:
            std::unordered_map<std::string, Parameter<T>*> m_named_params;
            Tensor<T> m_lr;
        public:
            Optimizer() = default;
            virtual ~Optimizer() = default;

            Device optim_device() {
                auto it = m_named_params.begin();
                return it->second->device();
            }

            void assert_valid_params() {
                if (std::ssize(m_named_params) == 0) {
                    throw std::runtime_error("Passed a vector of 0 parameters into the optimizer.");
                }
                Device target_device = this->optim_device();
                for (const auto& [name, p_ptr] : m_named_params) {
                    if (target_device != p_ptr->device()) {
                        throw std::runtime_error("All parameters passed into the optimizer must be on the same device.");
                    }
                }
            }

            // returns state dict moved to a specified device
            virtual std::unordered_map<std::string, Tensor<T>> state_dict([[maybe_unused]] Device device) {
                return {};
            }

            // loads state dict into the same device parameters are on
            virtual void load_state_dict(std::unordered_map<std::string, Tensor<T>> state_dict) {
                if (!state_dict.empty()) {
                    std::cout << "Invoked non-empty load_state_dict() on a stateless optimizer" << std::endl;
                }
            }

            // used to move different device param pointers into the optimizer, therefore updating its device
            void update_param_ptrs(std::unordered_map<std::string, Parameter<T>*> new_param_ptrs) {
                m_named_params = std::move(new_param_ptrs);
                this->assert_valid_params();
            }

            // used to move all states / metadata into the same device params are on
            virtual void sync_state_param_device() = 0;

            void update_lr(Tensor<T> lr) {
                m_lr = std::move(lr);
            }
    };
}