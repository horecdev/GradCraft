#pragma once

#include "gradc/nn/base/parameter.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class Optimizer {
        protected:
            std::vector<Parameter<T>*> m_params;
            std::unordered_map<Parameter<T>*, std::string> m_param_names;
        public:
            Optimizer() = default;
            virtual ~Optimizer() = default;

            virtual std::unordered_map<std::string, Tensor<T>> state_dict(Device device) {
                return {};
            }

            virtual void load_state_dict(std::unordered_map<std::string, Tensor<T>> state_dict) {
                if (!state_dict.empty()) {
                    std::cout << "Invoked non-empty load_state_dict() on a stateless optimizer" << std::endl;
                }
            }

            virtual void to(Device device) { // nothing to move
                return;
            }

            void assert_valid_params() {
                if (std::ssize(m_params) == 0) {
                    throw std::runtime_error("Passed a vector of 0 parameters into the optimizer.");
                }
                Device target_device = m_params[0]->device();
                for (Parameter<T>* p_ptr : m_params) {
                    if (target_device != p_ptr->device()) {
                        throw std::runtime_error("All parameters passed into the optimizer must be on the same device.");
                    }
                }
            }
    };
}