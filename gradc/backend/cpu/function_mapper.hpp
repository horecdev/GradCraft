#pragma once
#include "math_functors.hpp"
#include "../op_types.hpp"
#include <stdexcept>

namespace gradc::cpu_mapper {

    template <typename T, typename F>
    inline void map_boop(BinaryOp op, F&& callback) {
        // callback is a lambda taking a functor as param. You call lambda here. Lambda calls
        // the BOOP function with its parameter functor. Also bc of [&] it has access to local variables
        // You call this function (cpu_mapper::map_boop(op, lambda(auto functor)))
        // This function calls the lambda passing the right functor. Lambda calls the function with the right functor.
        switch (op) {
            case BinaryOp::Add:      callback(cpu_functors::BOOP::Add<T>()); break;
            case BinaryOp::Sub:      callback(cpu_functors::BOOP::Sub<T>()); break;
            case BinaryOp::Mul:      callback(cpu_functors::BOOP::Mul<T>()); break;
            case BinaryOp::Div:      callback(cpu_functors::BOOP::Div<T>()); break;
            case BinaryOp::EqMask:   callback(cpu_functors::BOOP::EqMask<T>()); break;
            case BinaryOp::BReLU:    callback(cpu_functors::BOOP::BReLU<T>()); break;
            case BinaryOp::BSigmoid: callback(cpu_functors::BOOP::BSigmoid<T>()); break;
            case BinaryOp::BTanH:    callback(cpu_functors::BOOP::BTanH<T>()); break;
            case BinaryOp::BSiLU:    callback(cpu_functors::BOOP::BSiLU<T>()); break;
            case BinaryOp::BGeLU:    callback(cpu_functors::BOOP::BGeLU<T>()); break;
            case BinaryOp::BExp:     callback(cpu_functors::BOOP::BExp<T>()); break;
            case BinaryOp::BLog:     callback(cpu_functors::BOOP::BLog<T>()); break;
            case BinaryOp::BSin:     callback(cpu_functors::BOOP::BSin<T>()); break;
            case BinaryOp::BCos:     callback(cpu_functors::BOOP::BCos<T>()); break;
            case BinaryOp::BSquare:  callback(cpu_functors::BOOP::BSquare<T>()); break;
            case BinaryOp::BSqrt:    callback(cpu_functors::BOOP::BSqrt<T>()); break;
            default: throw std::runtime_error("Unsupported BOOP in CPU mapper.");
        }
    }

    template <typename T, typename F>
    inline void map_bip(BinaryOpInPlace op, F&& callback) {
        switch (op) {
            case BinaryOpInPlace::Add:  callback(cpu_functors::BIP::Add<T>()); break;
            case BinaryOpInPlace::Sub:  callback(cpu_functors::BIP::Sub<T>()); break;
            case BinaryOpInPlace::ISub: callback(cpu_functors::BIP::ISub<T>()); break;
            case BinaryOpInPlace::Mul:  callback(cpu_functors::BIP::Mul<T>()); break;
            case BinaryOpInPlace::Div:  callback(cpu_functors::BIP::Div<T>()); break;
            case BinaryOpInPlace::IDiv: callback(cpu_functors::BIP::IDiv<T>()); break;
            default: throw std::runtime_error("Unsupported BIP in CPU mapper.");
        }
    }
    
    template <typename T, typename F>
    inline void map_uoop(UnaryOp op, F&& callback) { 
        switch (op) {
            case UnaryOp::Identity: callback(cpu_functors::UOOP::Identity<T>()); break;
            case UnaryOp::Neg:      callback(cpu_functors::UOOP::Neg<T>()); break;
            case UnaryOp::ReLU:     callback(cpu_functors::UOOP::ReLU<T>()); break;
            case UnaryOp::Sigmoid:  callback(cpu_functors::UOOP::Sigmoid<T>()); break;
            case UnaryOp::TanH:     callback(cpu_functors::UOOP::TanH<T>()); break;
            case UnaryOp::SiLU:     callback(cpu_functors::UOOP::SiLU<T>()); break;
            case UnaryOp::GeLU:     callback(cpu_functors::UOOP::GeLU<T>()); break;
            case UnaryOp::Exp:      callback(cpu_functors::UOOP::Exp<T>()); break;
            case UnaryOp::Log:      callback(cpu_functors::UOOP::Log<T>()); break;
            case UnaryOp::Sin:      callback(cpu_functors::UOOP::Sin<T>()); break;
            case UnaryOp::Cos:      callback(cpu_functors::UOOP::Cos<T>()); break;
            case UnaryOp::Square:   callback(cpu_functors::UOOP::Square<T>()); break;
            case UnaryOp::Sqrt:     callback(cpu_functors::UOOP::Sqrt<T>()); break;
            default: throw std::runtime_error("Unsupported UOOP in CPU mapper.");
        }
    }

    template <typename T, typename F>
    inline void map_uip(UnaryOpInPlace op, F&& callback) {
        switch (op) {
            case UnaryOpInPlace::Neg:      callback(cpu_functors::UIP::Neg<T>()); break;
            case UnaryOpInPlace::ReLU:     callback(cpu_functors::UIP::ReLU<T>()); break;
            case UnaryOpInPlace::Sigmoid:  callback(cpu_functors::UIP::Sigmoid<T>()); break;
            case UnaryOpInPlace::TanH:     callback(cpu_functors::UIP::TanH<T>()); break;
            case UnaryOpInPlace::SiLU:     callback(cpu_functors::UIP::SiLU<T>()); break;
            case UnaryOpInPlace::GeLU:     callback(cpu_functors::UIP::GeLU<T>()); break;
            case UnaryOpInPlace::Exp:      callback(cpu_functors::UIP::Exp<T>()); break;
            case UnaryOpInPlace::Log:      callback(cpu_functors::UIP::Log<T>()); break;
            case UnaryOpInPlace::Sin:      callback(cpu_functors::UIP::Sin<T>()); break;
            case UnaryOpInPlace::Cos:      callback(cpu_functors::UIP::Cos<T>()); break;
            case UnaryOpInPlace::Square:   callback(cpu_functors::UIP::Square<T>()); break;
            case UnaryOpInPlace::Sqrt:     callback(cpu_functors::UIP::Sqrt<T>()); break;
            default: throw std::runtime_error("Unsupported UIP in CPU mapper.");
        }
    }

    template <typename T, typename F>
    inline void map_red(ReduceOp op, F&& callback) {
        switch (op) {
            case ReduceOp::Sum: callback(cpu_functors::RED::Sum<T>(), T(0)); break;
            case ReduceOp::Max: callback(cpu_functors::RED::Max<T>(), std::numeric_limits<T>::lowest()); break;
            case ReduceOp::Min: callback(cpu_functors::RED::Min<T>(), std::numeric_limits<T>::max()); break;
            default: throw std::runtime_error("Unsupported ReduceOp in CPU mapper.");
        }
    }

    template <typename T, typename F>
    inline void map_argextr(ArgExtrOp op, F&& callback) {
        switch (op) {
            case ArgExtrOp::ArgMax: callback(cpu_functors::ARGEXTR::ArgMax<T>(), std::numeric_limits<T>::lowest()); break;
            case ArgExtrOp::ArgMin: callback(cpu_functors::ARGEXTR::ArgMin<T>(), std::numeric_limits<T>::max()); break;
            default: throw std::runtime_error("Unsupported ArgExtrOp in CPU mapper.");
        }
    }
}