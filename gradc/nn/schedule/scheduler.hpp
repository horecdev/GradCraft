#pragma once

#include "../optim/optimizer.hpp"
#include <numbers>

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

            virtual void load_state_dict(const std::unordered_map<std::string, T>& state) {
                m_t = static_cast<int64_t>(state["m_t"]);
                m_lr = state["m_lr"];
                
                // sync to optimizer
                m_optimizer->update_lr(m_lr);
            }
    };

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
                this->m_t++;
                
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
                Scheduler<T>::load_state_dict(state); // load the base
                
                m_max_lr = state["max_lr"];
                m_min_lr = state["min_lr"];
                m_warmup_steps = static_cast<int64_t>(state["warmup_steps"]);
                m_max_steps = static_cast<int64_t>(state["max_steps"]);
            }
    };


}