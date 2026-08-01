#pragma once

#include <cuda_runtime.h>

namespace gradc::cuda_functors {
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
        struct BReLU {
            __device__ T operator()(T grad, T value) const {
                return value > 0 ? grad : 0;
            }
        };

        template <typename T>
        struct EqMask {
            __device__ T operator()(T a, T b) const {
                return static_cast<T>(a == b);
            }
        };

        template <typename T>
        struct BSigmoid {
            __device__ T operator()(T grad, T sig_val) const {
                return grad * (sig_val) * (static_cast<T>(1.0) - sig_val);
            }
        };

        template <typename T>
        struct BTanH {
            __device__ T operator()(T grad, T tanh_val) const {
                return grad * (static_cast<T>(1.0) - tanh_val * tanh_val);
            }
        };
    }

    namespace BIP {
        template <typename T>
        struct Add {
            __device__ void operator()(T& a, T b) const {
                a += b;
            }
        };

        template <typename T>
        struct Sub {
            __device__ void operator()(T& a, T b) const {
                a -= b;
            }
        };

        template <typename T>
        struct ISub {
            __device__ void operator()(T& to_be_subbed, T main) const {
                to_be_subbed = -to_be_subbed + main;
            }
        };

        template <typename T>
        struct Mul {
            __device__ void operator()(T& a, T b) const {
                a *= b;
            }
        };

        template <typename T>
        struct Div {
            __device__ void operator()(T& a, T b) const {
                a /= b;
            }
        };

        template <typename T>
        struct IDiv {
            __device__ void operator()(T& divisor, T main) const {
                divisor = main / divisor;
            }
        };
    }

    namespace UOOP {
        template <typename T>
        struct Identity {
            __device__ T operator()(T a) const {
                return a;
            }
        };

        template <typename T>
        struct Neg {
            __device__ T operator()(T a) const {
                return -a;
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
                if constexpr (std::is_same_v<T, float>) {
                    return expf(a);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return exp(a);
                }
                else {
                    __trap(); // this shit should never be invoked (exponentiating a tensor of int type into int type)
                    return T(0);
                }
            }
        };

        template <typename T>
        struct Log {
            __device__ T operator()(T a) const {
                if constexpr (std::is_same_v<T, float>) {
                    return logf(a);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return log(a);
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct Sigmoid {
            __device__ T operator()(T a) const {
                if constexpr (std::is_same_v<T, float>) {
                    return 1.0f / (1.0f + expf(-a));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return 1.0 / (1.0 + exp(-a));
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct TanH {
            __device__ T operator()(T a) const {
                if constexpr (std::is_same_v<T, float>) {
                    return tanhf(a);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return tanh(a);
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename InT, typename OutT>
        struct Cast {
            __device__ OutT operator()(InT a) const {
                return static_cast<OutT>(a);
            }
        };
    }

    namespace UIP {
        template <typename T>
        struct Neg {
            __device__ void operator()(T& a) const {
                a = -a;
            }
        };

        template <typename T>
        struct ReLU {
            __device__ void operator()(T& a) const {
                a = a > 0 ? a : 0;
            }
        };

        template <typename T>
        struct Exp {
            __device__ void operator()(T& a) const {
                if constexpr (std::is_same_v<T, float>) {
                    a = expf(a);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    a = exp(a);
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct Log {
            __device__ void operator()(T& a) const {
                if constexpr (std::is_same_v<T, float>) {
                    a = logf(a);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    a = log(a);
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct Sigmoid {
            __device__ void operator()(T& a) const {
                if constexpr (std::is_same_v<T, float>) {
                    a = 1.0f / (1.0f + expf(-a));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    a = 1.0 / (1.0 + exp(-a));
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct TanH {
            __device__ void operator()(T& a) const {
                if constexpr (std::is_same_v<T, float>) {
                    a = tanhf(a);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    a = tanh(a);
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };
    }

    namespace RED {
        template <typename T>
        struct Sum {
            __device__ T operator()(T a, T b) const {
                return a + b;
            }

            __device__ void atomic(T* adress, T val) const {
                if constexpr (std::is_same_v<T, int64_t>) {
                    atomicAdd((unsigned long long int*)adress, val);
                }
                else {
                    atomicAdd(adress, val);
                }  
            }
        };

        template <typename T>
        struct Max {
            __device__ T operator()(T a, T b) const {
                return max(a, b);
            }

            __device__ void atomic(T* adress, T val) const {
                if constexpr (std::is_integral_v<T>) {
                    atomicMax(adress, val);
                }
                
                else if constexpr (std::is_same_v<T, float>) {
                    int* adress_as_int = (int*)adress;
                    int old = *adress_as_int; // what was in the adress when we entered the function
                    int assumed;

                    do {
                        assumed = old; // we assume what is in memory is still our adress

                        if (__int_as_float(assumed) >= val) { // perform a check here
                            break; // if what is now sitting in memory is bigger, skip
                        }

                        old = atomicCAS(adress_as_int, assumed, __float_as_int(val)); // CAS doesnt support integers either
                        // create a new expected value to compare against. If any of those was bigger than our number, break
                        // Why? Say current number at adress was 5. Our thread read it, and some other too. The other was bigger, so it overwrote it to 10. Ours is 7.
                        // If you overwrote either way, youre cooked.
                    } while (assumed != old);
                }

                else if constexpr (std::is_same_v<T, double>) {
                    unsigned long long int* adress_as_ull = (unsigned long long int*)adress; // unsigned so we take raw bytes
                    unsigned long long int old = *adress_as_ull; // old is smth like 13 quintillion because leftmost bit is signed. But CAS only compares bytes (same or not?)
                    unsigned long long int assumed;

                    do {
                        assumed = old;
                        if (__longlong_as_double(assumed) >= val) { // compare doubles, not long long
                            break;
                        }

                        old = atomicCAS(adress_as_ull, assumed, __double_as_longlong(val));

                    } while (assumed != old);
                }
            }
        };

        template <typename T>
        struct Min {
            __device__ T operator()(T a, T b) const {
                return min(a, b);
            }

            __device__ void atomic(T* adress, T val) const {
                if constexpr (std::is_integral_v<T>) {
                    atomicMax(adress, val);
                }
                
                else if constexpr (std::is_same_v<T, float>) {
                    int* adress_as_int = (int*)adress;
                    int old = *adress_as_int;
                    int assumed;

                    do {
                        assumed = old;

                        if (__int_as_float(assumed) <= val) {
                            break; 
                        }

                        old = atomicCAS(adress_as_int, assumed, __float_as_int(val)); 

                    } while (assumed != old);
                }

                else if constexpr (std::is_same_v<T, double>) {
                    unsigned long long int* adress_as_ull = (unsigned long long int*)adress;
                    unsigned long long int old = *adress_as_ull;
                    unsigned long long int assumed;

                    do {
                        assumed = old;
                        if (__longlong_as_double(assumed) <= val) {
                            break;
                        }

                        old = atomicCAS(adress_as_ull, assumed, __double_as_longlong(val));

                    } while (assumed != old);
                }
            }
        };

        
    }

    namespace ARGEXTR {
        template <typename T>
        struct ArgMax {
            __device__ bool operator()(T new_val, T current_max) {
                return new_val > current_max;
            }
        };

        template <typename T>
        struct ArgMin {
            __device__ bool operator()(T new_val, T current_min) {
                return new_val < current_min;
            }
        };
    }
}

