#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gradc::cpu_functors {
    namespace BOOP {
        template <typename T>
        struct Add {
            T operator()(T a, T b) const {
                return a + b;
            }
        };

        template <typename T>
        struct Sub {
            T operator()(T a, T b) const {
                return a - b;
            }
        };

        template <typename T>
        struct Mul {
            T operator()(T a, T b) const {
                return a * b;
            }
        };

        template <typename T>
        struct Div {
            T operator()(T a, T b) const {
                return a / b;
            }
        };

        template <typename T>
        struct EqMask {
            T operator()(T a, T b) const {
                return static_cast<T>(a == b);
            }
        };

        template <typename T>
        struct BExp {
            T operator()(T grad, T x) const {
                return grad * static_cast<T>(std::exp(x));
            }
        };

        template <typename T>
        struct BLog {
            T operator()(T grad, T x) const {
                return grad * (static_cast<T>(1.0) / x);
            }
        };

        template <typename T>
        struct BSin {
            T operator()(T grad, T x) const {
                return grad * static_cast<T>(std::cos(x));
            }
        };

        template <typename T>
        struct BCos {
            T operator()(T grad, T x) const {
                return -(grad * static_cast<T>(std::sin(x)));
            }
        };

        template <typename T>
        struct BSquare {
            T operator()(T grad, T x) const {
                return grad * 2 * x;
            }
        };

        template <typename T>
        struct BSqrt {
            T operator()(T grad, T x) const {
                return static_cast<T>(0.5) / static_cast<T>(std::sqrt(x)) * grad;
            }
        };

        template <typename T>
        struct BReLU {
            T operator()(T grad, T value) const {
                return value > 0 ? grad : 0;
            }
        };

        template <typename T>
        struct BSigmoid {
            T operator()(T grad, T x) const {
                T sig_val = static_cast<T>(1.0) / (static_cast<T>(1.0) + static_cast<T>(std::exp(-x)));
                return grad * (sig_val) * (static_cast<T>(1.0) - sig_val);
            }
        };

        template <typename T>
        struct BTanH {
            T operator()(T grad, T x) const {
                T tanh_val = static_cast<T>(std::tanh(x));
                return grad * (static_cast<T>(1.0) - tanh_val * tanh_val);
            }
        };
        
        template <typename T>
        struct BSiLU {
            T operator()(T grad, T x) const {
                T sig_val = static_cast<T>(1.0) / (static_cast<T>(1.0) + static_cast<T>(std::exp(-x)));
                return grad * (sig_val + x * sig_val * (1 - sig_val));
            }
        };  

        template <typename T>
        struct BGeLU {
            T operator()(T grad, T x) const {
                T alpha = static_cast<T>(0.7978845608);
                T beta = static_cast<T>(0.044715);
                T x_sq = x * x;
                T x_cu = x_sq * x;
                T u = alpha * (x + beta * x_cu);
                T t = static_cast<T>(std::tanh(u));
                T u_prime = alpha * (static_cast<T>(1.0) + static_cast<T>(3.0) * beta * x_sq);
                T local_grad = static_cast<T>(0.5) * (static_cast<T>(1.0) + t) + static_cast<T>(0.5) * x * (static_cast<T>(1.0) - t * t) * u_prime;
                return grad * local_grad;
            }
        };  
    }

    namespace BIP {
        template <typename T>
        struct Add {
            void operator()(T& a, T b) const {
                a += b;
            }
        };

        template <typename T>
        struct Sub {
            void operator()(T& a, T b) const {
                a -= b;
            }
        };

        template <typename T>
        struct ISub {
            void operator()(T& to_be_subbed, T main) const {
                to_be_subbed = -to_be_subbed + main;
            }
        };

        template <typename T>
        struct Mul {
            void operator()(T& a, T b) const {
                a *= b;
            }
        };

        template <typename T>
        struct Div {
            void operator()(T& a, T b) const {
                a /= b;
            }
        };

        template <typename T>
        struct IDiv {
            void operator()(T& divisor, T main) const {
                divisor = main / divisor;
            }
        };

        template <typename T>
        struct GrThan {
            void operator()(T& element, T value) const {
                element = element > value ? static_cast<T>(1.0) : static_cast<T>(0.0);
            }
        };
    }

    namespace UOOP {
        template <typename T>
        struct Identity {
            T operator()(T x) const {
                return x;
            }
        };

        template <typename T>
        struct Neg {
            T operator()(T x) const {
                return -x;
            }
        };

        template <typename T>
        struct ReLU {
            T operator()(T x) const {
                return x > 0 ? x : 0;
            }
        };

        template <typename T>
        struct Exp {
            T operator()(T x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return static_cast<T>(std::exp(x));
                }
                else {
                    throw std::runtime_error("Exp CPU UOOP functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Log {
            T operator()(T x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return static_cast<T>(std::log(x));
                }
                else {
                    throw std::runtime_error("Log CPU UOOP functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Sin {
            T operator()(T x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return static_cast<T>(std::sin(x));
                }
                else {
                    throw std::runtime_error("Sin CPU UOOP functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Cos {
            T operator()(T x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return static_cast<T>(std::cos(x));
                }
                else {
                    throw std::runtime_error("Cos CPU UOOP functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Square {
            T operator()(T x) const {
                return x * x;
            }
        };

        template <typename T>
        struct Sqrt {
            T operator()(T x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return std::sqrt(x);
                }
                else {
                    throw std::runtime_error("Sqrt CPU UOOP functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Sigmoid {
            T operator()(T x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
                }
                else {
                    throw std::runtime_error("Sigmoid CPU UOOP functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct TanH {
            T operator()(T x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return static_cast<T>(std::tanh(x));
                }
                else {
                    throw std::runtime_error("tanh CPU UOOP functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct SiLU {
            T operator()(T x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return x * (static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x)));
                }
                else {
                    throw std::runtime_error("SiLU CPU UOOP functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct GeLU {
            T operator()(T x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    T alpha = static_cast<T>(0.7978845608);
                    T beta = static_cast<T>(0.044715);
                    return x * static_cast<T>(0.5) * (static_cast<T>(1.0) + std::tanh(alpha * (x + beta * x * x * x)));
                }
                else {
                    throw std::runtime_error("GeLU CPU UOOP functor running on non-floating.");
                }
            }
        };

        template <typename InT, typename OutT>
        struct Cast {
            OutT operator()(InT x) const {
                return static_cast<OutT>(x);
            }
        };
    }

    namespace UIP {
        template <typename T>
        struct Neg {
            void operator()(T& x) const {
                x = -x;
            }
        };

        template <typename T>
        struct ReLU {
            void operator()(T& x) const {
                x = x > 0 ? x : 0;
            }
        };

        template <typename T>
        struct Exp {
            void operator()(T& x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    x = static_cast<T>(std::exp(x));
                }
                else {
                    throw std::runtime_error("UIP Exp CPU functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Log {
            void operator()(T& x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    x = static_cast<T>(std::log(x));
                }
                else {
                    throw std::runtime_error("UIP Log CPU functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Sin {
            void operator()(T& x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    x = static_cast<T>(std::sin(x));
                }
                else {
                    throw std::runtime_error("UIP Sin CPU functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Cos {
            void operator()(T& x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    x = static_cast<T>(std::cos(x));
                }
                else {
                    throw std::runtime_error("UIP Cos CPU functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Square {
            void operator()(T& x) const {
                x = x * x;
            }
        };

        template <typename T>
        struct Sqrt {
            void operator()(T& x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    x = static_cast<T>(std::cos(x));
                }
                else {
                    throw std::runtime_error("UIP Sqrt CPU functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Sigmoid {
            void operator()(T& x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    x = static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
                }
                else {
                    throw std::runtime_error("UIP Sigmoid CPU functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct TanH {
            void operator()(T& x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    x = static_cast<T>(std::tanh(x));
                }
                else {
                    throw std::runtime_error("UIP tanh CPU functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct SiLU {
            void operator()(T& x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    x = x * (static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x)));
                }
                else {
                    throw std::runtime_error("SiLU CPU UIP functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct GeLU {
            void operator()(T& x) const {
                if constexpr (std::is_floating_point_v<T>) {
                    T alpha = static_cast<T>(0.7978845608);
                    T beta = static_cast<T>(0.044715);
                    x = x * static_cast<T>(0.5) * (static_cast<T>(1.0) + std::tanh(alpha * (x + beta * x * x * x)));
                }
                else {
                    throw std::runtime_error("GeLU CPU UIP functor running on non-floating.");
                }
            }
        };
    }

    namespace RED {
        template <typename T>
        struct Sum {
            T operator()(T a, T b) const {
                return a + b;
            }
        };

        template <typename T>
        struct Max {
            T operator()(T a, T b) const {
                return std::max(a, b);
            }
        };

        template <typename T>
        struct Min {
            T operator()(T a, T b) const {
                return std::min(a, b);
            }
        };
    }

    namespace ARGEXTR {
        template <typename T>
        struct ArgMax {
            bool operator()(T new_val, T current_max) {
                return new_val > current_max;
            }
        };

        template <typename T>
        struct ArgMin {
            bool operator()(T new_val, T current_min) {
                return new_val < current_min;
            }
        };
    }
}

