#pragma once

enum class UnaryOp {Identity, Neg, ReLU, Exp, Log, Sigmoid, TanH, SiLU, GeLU, Cast};
enum class UnaryOpInPlace {ReLU, Neg, Exp, Log, Sigmoid, TanH, SiLU, GeLU};
enum class BinaryOp {Add, Sub, Mul, Div, MatMul, EqMask, BReLU, BSigmoid, BTanH, BSiLU, BGeLU, BExp, BLog};
enum class BinaryOpInPlace {Add, Sub, ISub, Mul, Div, IDiv, MatMul};
enum class ReduceOp {Sum, Mean, Max, Min};
enum class ArgExtrOp {ArgMax, ArgMin};