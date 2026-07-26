#pragma once

#include <cuda_runtime.h>

namespace gradc::functors {
    namespace BOOP {
        template <typename T>
        struct Add {
            __device__ T operator()(T a, T b) const {
                return a + b;
            }
        };

        template <typename T>
        struct Sub {
            __device__ T operator()(T a, T b) const {
                return a - b;
            }
        };

        template <typename T>
        struct Mul {
            __device__ T operator()(T a, T b) const {
                return a * b;
            }
        };

        template <typename T>
        struct Div {
            __device__ T operator()(T a, T b) const {
                return a / b;
            }
        };

        template <typename T>
        struct ReLUBackward {
            __device__ T operator()(T grad, T value) const {
                return value > 0 ? grad : 0;
            }
        };
    }

    namespace BIP {
        template <typename T>
        struct Add {
            __device__ T operator()(T& a, T b) const {
                a += b;
            }
        };

        template <typename T>
        struct Sub {
            __device__ T operator()(T& a, T b) const {
                a -= b;
            }
        };

        template <typename T>
        struct Mul {
            __device__ T operator()(T& a, T b) const {
                a *= b;
            }
        };

        template <typename T>
        struct Div {
            __device__ T operator()(T& a, T b) const {
                a /= b;
            }
        };
    }

    namespace UOP {
        template <typename T>
        struct Identity {
            __device__ T operator()(T a) const {
                return a;
            }
        };

        template <typename InT, typename OutT>
        struct Cast {
            __device__ OutT operator()(InT a) const {
                return static_cast<OutT>(a);
            }
        };

        template <typename T>
        struct ReLU {
            __device__ T operator()(T a) const {
                return a > 0 ? a : 0;
            }
        };

        template <typename T>
        struct Exp {
            __device__ T operator()(T a) const {
                return exp(a);
            }
        };

        template <typename T>
        struct Log {
            __device__ T operator()(T a) const {
                return log(a);
            }
        };
    }

    namespace UIP {
        template <typename T>
        struct ReLU {
            __device__ T operator()(T& a) const {
                a = a > 0 ? a : 0;
            }
        };

        template <typename T>
        struct Log {
            __device__ T operator()(T& a) const {
                a = log(a);
            }
        };

        template <typename T>
        struct Exp {
            __device__ T operator()(T& a) const {
                a = exp(a);
            }
        };
    }
    
}