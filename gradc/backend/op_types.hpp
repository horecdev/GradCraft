#pragma once

enum class UnaryOp {Identity, ReLU, Sigmoid, Exp, Log, Cast};
enum class UnaryOpInPlace {ReLU, Sigmoid, Exp, Log};
enum class BinaryOp {Add, Sub, Mul, Div, MatMul, ReLUBackward, EqMask};
enum class BinaryOpInPlace {Add, Sub, Mul, Div, MatMul};
enum class ReduceOp {Sum, Mean, Max, Min};