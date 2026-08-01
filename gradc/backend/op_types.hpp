#pragma once

enum class UnaryOp {Identity, Neg, ReLU, Exp, Log, Sigmoid, TanH, Cast};
enum class UnaryOpInPlace {ReLU, Neg, Exp, Log, Sigmoid, TanH};
enum class BinaryOp {Add, Sub, Mul, Div, MatMul, ReLUBackward, EqMask};
enum class BinaryOpInPlace {Add, Sub, ISub, Mul, Div, IDiv, MatMul};
enum class ReduceOp {Sum, Mean, Max, Min};
enum class ArgExtrOp {ArgMax, ArgMin};