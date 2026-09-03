#pragma once

#include "../optim/optimizer.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class Scheduler {
        protected:
            Optimizer<T>* m_optimizer;
        public:
            int64_t m_t;
            T m_lr;

            Scheduler(Optimizer<T>* optimizer, T init_lr) : m_optimizer(optimizer), m_t(0), m_lr(init_lr) {}

            virtual void step() = 0;

            virtual std::unordered_map<std::string, T> state_dict() const { // always on the CPU
                std::unordered_map<std::string, T> result;
                result["m_t"] = static_cast<T>(m_t);
                result["m_lr"] = m_lr;
            }

            virtual void load_state_dict(const std::unordered_map<std::string, T>& state) { // optimizer MUST have lr allocated.
                m_t = static_cast<int64_t>(state["m_t"]);
                m_lr = state["m_lr"];
                
                // sync to optimizer
                m_optimizer->update_lr(m_lr);
            }
    };

    

}