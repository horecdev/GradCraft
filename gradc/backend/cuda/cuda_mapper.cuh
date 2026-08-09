#pragma once
#include "gradc/backend/cuda/math_functors.cuh"
#include "gradc/backend/op_types.hpp"
#include <stdexcept>

namespace gradc::cuda_mapper {


    template <typename T, typename F>
    inline void map_boop(BinaryOp op, F&& callback) {
        switch (op) {
            case BinaryOp::Add:      callback(cuda_functors::BOOP::Add<T>()); break;
            case BinaryOp::Sub:      callback(cuda_functors::BOOP::Sub<T>()); break;
            case BinaryOp::Mul:      callback(cuda_functors::BOOP::Mul<T>()); break;
            case BinaryOp::Div:      callback(cuda_functors::BOOP::Div<T>()); break;
            case BinaryOp::EqMask:   callback(cuda_functors::BOOP::EqMask<T>()); break;
            case BinaryOp::BReLU:    callback(cuda_functors::BOOP::BReLU<T>()); break;
            case BinaryOp::BSigmoid: callback(cuda_functors::BOOP::BSigmoid<T>()); break;
            case BinaryOp::BTanH:    callback(cuda_functors::BOOP::BTanH<T>()); break;
            case BinaryOp::BSiLU:    callback(cuda_functors::BOOP::BSiLU<T>()); break;
            case BinaryOp::BGeLU:    callback(cuda_functors::BOOP::BGeLU<T>()); break;
            case BinaryOp::BExp:     callback(cuda_functors::BOOP::BExp<T>()); break;
            case BinaryOp::BLog:     callback(cuda_functors::BOOP::BLog<T>()); break;
            case BinaryOp::BSin:     callback(cuda_functors::BOOP::BSin<T>()); break;
            case BinaryOp::BCos:     callback(cuda_functors::BOOP::BCos<T>()); break;
            case BinaryOp::BSquare:  callback(cuda_functors::BOOP::BSquare<T>()); break;
            case BinaryOp::BSqrt:    callback(cuda_functors::BOOP::BSqrt<T>()); break;
            default: throw std::runtime_error("Unsupported BOOP in CUDA mapper.");
        }
    }

    template <typename T, typename F>
    inline void map_bip(BinaryOpInPlace op, F&& callback) {
        switch (op) {
            case BinaryOpInPlace::Add:  callback(cuda_functors::BIP::Add<T>()); break;
            case BinaryOpInPlace::Sub:  callback(cuda_functors::BIP::Sub<T>()); break;
            case BinaryOpInPlace::ISub: callback(cuda_functors::BIP::ISub<T>()); break;
            case BinaryOpInPlace::Mul:  callback(cuda_functors::BIP::Mul<T>()); break;
            case BinaryOpInPlace::Div:  callback(cuda_functors::BIP::Div<T>()); break;
            case BinaryOpInPlace::IDiv: callback(cuda_functors::BIP::IDiv<T>()); break;
            default: throw std::runtime_error("Unsupported BIP in CUDA mapper.");
        }
    }

    template <typename T, typename F>
    inline void map_uoop(UnaryOp op, F&& callback) {
        switch (op) {
            case UnaryOp::Identity: callback(cuda_functors::UOOP::Identity<T>()); break;
            case UnaryOp::Neg:      callback(cuda_functors::UOOP::Neg<T>()); break;
            case UnaryOp::ReLU:     callback(cuda_functors::UOOP::ReLU<T>()); break;
            case UnaryOp::Sigmoid:  callback(cuda_functors::UOOP::Sigmoid<T>()); break;
            case UnaryOp::TanH:     callback(cuda_functors::UOOP::TanH<T>()); break;
            case UnaryOp::SiLU:     callback(cuda_functors::UOOP::SiLU<T>()); break;
            case UnaryOp::GeLU:     callback(cuda_functors::UOOP::GeLU<T>()); break;
            case UnaryOp::Exp:      callback(cuda_functors::UOOP::Exp<T>()); break;
            case UnaryOp::Log:      callback(cuda_functors::UOOP::Log<T>()); break;
            case UnaryOp::Sin:      callback(cuda_functors::UOOP::Sin<T>()); break;
            case UnaryOp::Cos:      callback(cuda_functors::UOOP::Cos<T>()); break;
            case UnaryOp::Square:   callback(cuda_functors::UOOP::Square<T>()); break;
            case UnaryOp::Sqrt:     callback(cuda_functors::UOOP::Sqrt<T>()); break;
            default: throw std::runtime_error("Unsupported UOOP in CUDA mapper.");
        }
    }

    template <typename T, typename F>
    inline void map_uip(UnaryOpInPlace op, F&& callback) {
        switch (op) {
            case UnaryOpInPlace::Neg:      callback(cuda_functors::UIP::Neg<T>()); break;
            case UnaryOpInPlace::ReLU:     callback(cuda_functors::UIP::ReLU<T>()); break;
            case UnaryOpInPlace::Sigmoid:  callback(cuda_functors::UIP::Sigmoid<T>()); break;
            case UnaryOpInPlace::TanH:     callback(cuda_functors::UIP::TanH<T>()); break;
            case UnaryOpInPlace::SiLU:     callback(cuda_functors::UIP::SiLU<T>()); break;
            case UnaryOpInPlace::GeLU:     callback(cuda_functors::UIP::GeLU<T>()); break;
            case UnaryOpInPlace::Exp:      callback(cuda_functors::UIP::Exp<T>()); break;
            case UnaryOpInPlace::Log:      callback(cuda_functors::UIP::Log<T>()); break;
            case UnaryOpInPlace::Sin:      callback(cuda_functors::UIP::Sin<T>()); break;
            case UnaryOpInPlace::Cos:      callback(cuda_functors::UIP::Cos<T>()); break;
            case UnaryOpInPlace::Square:   callback(cuda_functors::UIP::Square<T>()); break;
            case UnaryOpInPlace::Sqrt:     callback(cuda_functors::UIP::Sqrt<T>()); break;
            default: throw std::runtime_error("Unsupported UIP in CUDA mapper.");
        }
    }

    template <typename T, typename F>
    inline void map_red(ReduceOp op, F&& callback) {
        switch (op) {
            case ReduceOp::Sum: callback(cuda_functors::RED::Sum<T>(), T(0)); break;
            case ReduceOp::Max: callback(cuda_functors::RED::Max<T>(), std::numeric_limits<T>::lowest()); break;
            case ReduceOp::Min: callback(cuda_functors::RED::Min<T>(), std::numeric_limits<T>::max()); break;
            default: throw std::runtime_error("Unsupported ReduceOp in CUDA mapper.");
        }
    }

    template <typename T, typename F>
    inline void map_argextr(ArgExtrOp op, F&& callback) {
        switch (op) {
            case ArgExtrOp::ArgMax: callback(cuda_functors::ARGEXTR::ArgMax<T>(), std::numeric_limits<T>::lowest()); break;
            case ArgExtrOp::ArgMin: callback(cuda_functors::ARGEXTR::ArgMin<T>(), std::numeric_limits<T>::max()); break;
            default: throw std::runtime_error("Unsupported ArgExtrOp in CUDA mapper.");
        }
    }
}