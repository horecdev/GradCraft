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

        template <typename T>
        struct BSiLU {
            __device__ T operator()(T grad, T x) const {
                if constexpr (std::is_same_v<T, float>) {
                    T sig_val = static_cast<T>(1.0) / (static_cast<T>(1.0) + expf(-x));
                    return grad * (sig_val + x * sig_val * (1 - sig_val));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    T sig_val = static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x));
                    return grad * (sig_val + x * sig_val * (1 - sig_val));
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };  

        template <typename T>
        struct BGeLU {
            __device__ T operator()(T grad, T x) const {
                if constexpr (std::is_same_v<T, float>) {
                    T alpha = 0.7978845608;
                    T beta = 0.044715;
                    T x_sq = x * x;
                    T x_cu = x_sq * x;
                    T u = alpha * (x + beta * x_cu);
                    T t = tanh(u);
                    T u_prime = alpha * (static_cast<T>(1.0) + static_cast<T>(3.0) * beta * x_sq);
                    T local_grad = static_cast<T>(0.5) * (static_cast<T>(1.0) + t) + static_cast<T>(0.5) * x * (static_cast<T>(1.0) - t * t) * u_prime;
                    return grad * local_grad;
                }
                else if constexpr (std::is_same_v<T, double>) {
                    T alpha = 0.7978845608;
                    T beta = 0.044715;
                    T x_sq = x * x;
                    T x_cu = x_sq * x;
                    T u = alpha * (x + beta * x_cu);
                    T t = tanhf(u);
                    T u_prime = alpha * (static_cast<T>(1.0) + static_cast<T>(3.0) * beta * x_sq);
                    T local_grad = static_cast<T>(0.5) * (static_cast<T>(1.0) + t) + static_cast<T>(0.5) * x * (static_cast<T>(1.0) - t * t) * u_prime;
                    return grad * local_grad;
                }
                else {
                    __trap();
                    return T(0);
                }
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
            __device__ T operator()(T x) const {
                return x;
            }
        };

        template <typename T>
        struct Neg {
            __device__ T operator()(T x) const {
                return -x;
            }
        };

        template <typename T>
        struct ReLU {
            __device__ T operator()(T x) const {
                return x > 0 ? x : 0;
            }
        };

        template <typename T>
        struct Exp {
            __device__ T operator()(T x) const {
                if constexpr (std::is_same_v<T, float>) {
                    return expf(x);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return exp(x);
                }
                else {
                    __trap(); // this shit should never be invoked (exponentiating a tensor of int type into int type)
                    return T(0);
                }
            }
        };

        template <typename T>
        struct Log {
            __device__ T operator()(T x) const {
                if constexpr (std::is_same_v<T, float>) {
                    return logf(x);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return log(x);
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct Sigmoid {
            __device__ T operator()(T x) const {
                if constexpr (std::is_same_v<T, float>) {
                    return 1.0f / (1.0f + expf(-x));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return 1.0 / (1.0 + exp(-x));
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct TanH {
            __device__ T operator()(T x) const {
                if constexpr (std::is_same_v<T, float>) {
                    return tanhf(x);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return tanh(x);
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct SiLU {
            __device__ T operator()(T x) const {
                if constexpr (std::is_same_v<T, float>) {
                    return x * (static_cast<T>(1.0) / (static_cast<T>(1.0) + expf(-x)));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    return x * (static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x)));
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct GeLU {
            __device__ T operator()(T x) const {
                if constexpr (std::is_same_v<T, float>) {
                    T alpha = static_cast<T>(0.7978845608);
                    T beta = static_cast<T>(0.044715);
                    return x * static_cast<T>(0.5) * (static_cast<T>(1.0) + tanhf(alpha * (x + beta * x * x * x)));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    T alpha = static_cast<T>(0.7978845608);
                    T beta = static_cast<T>(0.044715);
                    return x * static_cast<T>(0.5) * (static_cast<T>(1.0) + tanh(alpha * (x + beta * x * x * x)));
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename InT, typename OutT>
        struct Cast {
            __device__ OutT operator()(InT x) const {
                return static_cast<OutT>(x);
            }
        };
    }

    namespace UIP {
        template <typename T>
        struct Neg {
            __device__ void operator()(T& x) const {
                x = -x;
            }
        };

        template <typename T>
        struct ReLU {
            __device__ void operator()(T& x) const {
                x = x > 0 ? x : 0;
            }
        };

        template <typename T>
        struct Exp {
            __device__ void operator()(T& x) const {
                if constexpr (std::is_same_v<T, float>) {
                    x = expf(x);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    x = exp(x);
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct Log {
            __device__ void operator()(T& x) const {
                if constexpr (std::is_same_v<T, float>) {
                    x = logf(x);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    x = log(x);
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct Sigmoid {
            __device__ void operator()(T& x) const {
                if constexpr (std::is_same_v<T, float>) {
                    x = 1.0f / (1.0f + expf(-x));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    x = 1.0 / (1.0 + exp(-x));
                }
                else {
                    __trap();
                    return T(0);
                }
            }
        };

        template <typename T>
        struct TanH {
            __device__ void operator()(T& x) const {
                if constexpr (std::is_same_v<T, float>) {
                    x = tanhf(x);
                }
                else if constexpr (std::is_same_v<T, double>) {
                    x = tanh(x);
                }
                else {
                    __trap();
                    x = T(0);
                }
            }
        };

        template <typename T>
        struct SiLU {
            __device__ T operator()(T& x) const {
                if constexpr (std::is_same_v<T, float>) {
                    x = x * (static_cast<T>(1.0) / (static_cast<T>(1.0) + expf(-x)));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    x = x * (static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-x)));
                }
                else {
                    __trap();
                    x = T(0);
                }
            }
        };

        template <typename T>
        struct GeLU {
            __device__ T operator()(T& x) const {
                if constexpr (std::is_same_v<T, float>) {
                    T alpha = static_cast<T>(0.7978845608);
                    T beta = static_cast<T>(0.044715);
                    x = x * static_cast<T>(0.5) * (static_cast<T>(1.0) + tanhf(alpha * (x + beta * x * x * x)));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    T alpha = static_cast<T>(0.7978845608);
                    T beta = static_cast<T>(0.044715);
                    x = x * static_cast<T>(0.5) * (static_cast<T>(1.0) + tanh(alpha * (x + beta * x * x * x)));
                }
                else {
                    __trap();
                    x = T(0);
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

