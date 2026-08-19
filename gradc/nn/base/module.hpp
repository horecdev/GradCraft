#pragma once

#include "parameter.hpp"

namespace gradc {
    template <typename T>
    class Module {
        protected:
            bool m_is_training;
            std::vector<std::pair<std::string, Module<T>*>> m_submodules;
            std::vector<std::pair<std::string, Parameter<T>*>> m_parameters;
        public:
            Module();

            bool is_training() const {
                return m_is_training;
            }

            bool train() {
                m_is_training = true;
                for (const auto& [name, module] : m_submodules) {
                    module->train();
                }
            }

            bool eval() {
                m_is_training = false;
                for (const auto& [name, module] : m_submodules) {
                    module->eval();
                }
            }

            void register_module(std::string prefix, Module<T>* module) {
                m_submodules.push_back(std::make_pair(std::move(prefix), module));
            }

            void register_parameter(std::string prefix, Parameter<T>* param) {
                m_parameters.push_back(std::make_pair(std::move(prefix), param));
            } 
            
            std::vector<std::pair<std::string, Parameter<T>*>> named_parameters(std::string prefix) const {
                std::vector<std::pair<std::string, Parameter<T>*>> result;
                for (const auto& [param_name, param_ptr] : m_parameters) {
                    std::string full_param_name = prefix.empty() ? param_name : prefix + "." + param_name;
                    result.push_back(std::make_pair(full_param_name, param_ptr));
                }

                for (const auto& [module_name, module_ptr] : m_submodules) {
                    std::string full_module_prefix = prefix.empty() ? module_name : prefix + "." + module_name;
                    auto child_module_params = module_ptr->named_parameters(full_module_prefix);

                    result.insert(result.end(), child_module_params.begin(), child_module_params.end());
                }
            }

            std::vector<Parameter<T>*> parameters() {
                std::vector<Parameter<T>*> result;
                auto named_params = named_parameters();
                result.reserve(std::ssize(named_params));
                for (const auto& [param_name, param_ptr] : named_params) {
                    result.push_back(param_ptr);
                }
                return result;
            }

            std::unordered_map<std::string, Tensor<T>> state_dict(Device device = Device(DeviceType::CPU)) const {
                // takes any tensor you give it, does .to(), realizes it, detaches, saves it in the map.
                std::unordered_map<std::string, Tensor<T>> dict;
                auto named_params = named_parameters();
                for (const auto& [param_name, param_ptr] : named_params) {
                    Tensor<T> moved_tensor = param_ptr->tensor().to(device); 
                    moved_tensor.realize();
                    dict[param_name] = moved_tensor.detach();
                    // returned state dict is totally detached (requires_grad = false, no graph)
                }
                return dict;
            }

            void load_state_dict(const std::unordered_map<std::string, Tensor<T>>& state_dict) {
                // takes your map, does key search, runs .to() and detaches, moves the detached device-aware tensor into the param used across the network
                auto current_named_params = named_parameters();
                for (const auto& [param_name, param_ptr] : current_named_params) {
                    auto it = state_dict.find(param_name);
                    if (it != state_dict.end()) {
                        Tensor<T> moved_tensor = it->second.to(param_ptr->tensor().device());
                        moved_tensor.realize();
                        Tensor<T> leaf_tensor = moved_tensor.detach();
                        leaf_tensor.set_requires_grad(param_ptr->requires_grad()); // retain same requires_grad
                        param_ptr->tensor() = std::move(leaf_tensor);
                    } 
                    else {
                        std::cout << "Warning: '" + param_name + "' is in the model, but couldn't find its counterpart in state_dict during loading." << std::endl;
                    }
                }
            }

            void zero_grad() {
                auto params = parameters();
                for (Parameter<T>* p : params) {
                    p->zero_grad();
                }
            }

            // to() must EAGERLY MOVE MEMORY - you want your leaf nodes to remain leaf nodes.
            void to(Device device) {
                auto params = parameters();
                for (auto param_ptr : params) {
                    Tensor<T> moved = param_ptr->tensor().to(device);
                    moved.realize(); // force move across PCIe bus
                    Tensor<T> leaf_tensor = moved.detach();
                    leaf_tensor.set_requires_grad(param_ptr->requires_grad());
                    param_ptr->tensor() = std::move(leaf_tensor); // parameter has the detached tensor
                }
            }
    };
}