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
        struct ReLUBackward {
            T operator()(T grad, T value) const {
                return value > 0 ? grad : 0;
            }
        };

        template <typename T>
        struct EqMask {
            T operator()(T a, T b) const {
                return static_cast<T>(a == b);
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
    }

    namespace UOP {
        template <typename T>
        struct Identity {
            T operator()(T a) const {
                return a;
            }
        };

        template <typename T>
        struct ReLU {
            T operator()(T a) const {
                return a > 0 ? a : 0;
            }
        };

        template <typename T>
        struct Exp {
            T operator()(T a) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return static_cast<T>(std::exp(a));
                }
                else {
                    throw std::runtime_error("Exp CPU functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Log {
            T operator()(T a) const {
                if constexpr (std::is_floating_point_v<T>) {
                    return static_cast<T>(std::log(a));
                }
                else {
                    throw std::runtime_error("Log CPU functor running on non-floating.");
                }
            }
        };

        template <typename InT, typename OutT>
        struct Cast {
            OutT operator()(InT a) const {
                return static_cast<OutT>(a);
            }
        };
    }

    namespace UIP {
        template <typename T>
        struct ReLU {
            void operator()(T& a) const {
                a = a > 0 ? a : 0;
            }
        };

        template <typename T>
        struct Exp {
            void operator()(T& a) const {
                if constexpr (std::is_floating_point_v<T>) {
                    a = static_cast<T>(std::exp(a));
                }
                else {
                    throw std::runtime_error("UIP Exp CPU functor running on non-floating.");
                }
            }
        };

        template <typename T>
        struct Log {
            void operator()(T& a) const {
                if constexpr (std::is_floating_point_v<T>) {
                    a = static_cast<T>(std::log(a));
                }
                else {
                    throw std::runtime_error("UIP Log CPU functor running on non-floating.");
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

