#pragma once

#include <numbers>
#include "scheduler.hpp"

namespace gradc {
    template <typename T>
    requires std::is_floating_point_v<T>
    class CosineScheduler : public Scheduler<T> {
        private:
            T m_max_lr;
            T m_min_lr;
            int64_t m_warmup_steps;
            int64_t m_max_steps;

        public:
            CosineScheduler(Optimizer<T>* optimizer, T max_lr, T min_lr, int64_t warmup_steps, int64_t max_steps)
             : Scheduler<T>(optimizer, 0.0), /*lr set to 0 cuz warmup*/ m_max_lr(max_lr), m_min_lr(min_lr), m_warmup_steps(warmup_steps), m_max_steps(max_steps) {}

            void step() override {
                this->m_t++; // scheduler always does the step FIRST. Its at t=1. Then gives it to optimizer at step 0. It moves from 0 to 1. Both are at 1. Then sched moves from 1 to 2, etc.
                // when you LOAD: scheduler and opt are at t=2. First scheduler steps, gets lr for t=3, passes to optimizer to do step 3. Works fully.
                
                if (this->m_t <= m_warmup_steps) {
                    this->m_lr = m_max_lr * (static_cast<T>(this->m_t) / static_cast<T>(m_warmup_steps));
                } 
                else if (this->m_t >= m_max_steps) {
                    this->m_lr = m_min_lr;
                } 
                else {
                    T decay_ratio = static_cast<T>(this->m_t - m_warmup_steps) / static_cast<T>(m_max_steps - m_warmup_steps); // "percent done"
                    T coeff = static_cast<T>(0.5) * (static_cast<T>(1.0) + std::cos(std::numbers::pi_v<T> * decay_ratio)); // when "percent done" is 100%, cos is at -1 so its all 0. At 0% cos is 1, 0.5(1 + 1) = 1
                    this->m_lr = m_min_lr + coeff * (m_max_lr - m_min_lr); // when coeff = 1 then lr = max_lr, at coeff = 0 its min_lr
                }

                this->m_optimizer->update_lr(this->m_lr);
            }

            std::unordered_map<std::string, T> state_dict() const override {
                std::unordered_map<std::string, T> state = Scheduler<T>::state_dict(); // get the base from base class
                state["max_lr"] = m_max_lr;
                state["min_lr"] = m_min_lr;
                state["warmup_steps"] = static_cast<T>(m_warmup_steps);
                state["max_steps"] = static_cast<T>(m_max_steps);
                return state;
            }

            void load_state_dict(const std::unordered_map<std::string, T>& state) override {
                Scheduler<T>::load_state_dict(state); // load the base, push the lr inside
                
                m_max_lr = state["max_lr"];
                m_min_lr = state["min_lr"];
                m_warmup_steps = static_cast<int64_t>(state["warmup_steps"]);
                m_max_steps = static_cast<int64_t>(state["max_steps"]);
            }
    };
}

