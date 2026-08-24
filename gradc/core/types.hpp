#pragma once

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
#include <iostream>

namespace gradc {
    // CORE DTYPES
    enum class DType {Float32, Float64, Int32, Int64, Unknown};

    template <typename T>
    constexpr DType type_to_dtype() { // before signature: allows the compiler to figure out at compilation
        if constexpr (std::is_same_v<T, float>) return DType::Float32; // before if statement: deletes the line if false, keeps if true
        else if constexpr (std::is_same_v<T, double>) return DType::Float64;
        else if constexpr (std::is_same_v<T, int32_t>) return DType::Int32;
        else if constexpr (std::is_same_v<T, int64_t>) return DType::Int64;
        else return DType::Unknown;
    }

    template <typename T>
    constexpr bool is_supported_tensor_type_v = 
        std::is_same_v<T, int32_t> ||
        std::is_same_v<T, int64_t> ||
        std::is_same_v<T, float> ||
        std::is_same_v<T, double>;

    // DEVICES

    enum class DeviceType {CPU, CUDA};

    struct Device {
        DeviceType type;
        int32_t index;

        constexpr Device(DeviceType type = DeviceType::CPU, int32_t index = -1) : type(type), index(index) {}

        constexpr bool operator==(const Device& other) const {
            if (this->type == DeviceType::CPU) {
                return other.type == DeviceType::CPU;
            }
            else {
                return (type == other.type && index == other.index);
            }
        }

        constexpr bool operator!=(const Device& other) const {
            return !(*this == other);
        }

        friend std::ostream& operator<<(std::ostream& os, const Device& d) {
            if (d.type == DeviceType::CPU) {
                os << "CPU";
            }
            else if (d.type == DeviceType::CUDA) {
                os << "CUDA:" << d.index;
            }
            return os;
        }

        constexpr bool is_cpu() {
            return type == DeviceType::CPU;
        }

        constexpr bool is_cuda() {
            return type == DeviceType::CUDA;
        }
    };

    // INDEXING

    enum IndexType {Single, Range, All};

    struct Slice {
        std::optional<int64_t> start;
        std::optional<int64_t> stop;
        std::optional<int64_t> step;

        Slice(std::optional<int64_t> start = std::nullopt, std::optional<int64_t> stop = std::nullopt, std::optional<int64_t> step = std::nullopt) 
            : start(start), stop(stop), step(step) {}
    };

    struct AllSlice{};
    inline constexpr AllSlice _ = AllSlice(); // inline doesnt confuse linker if its #included in 2+ files.

    struct IndexDesc {
        IndexType m_type;
        int64_t m_single_val;
        Slice m_slice_val;

        IndexDesc(int64_t value) : m_type(IndexType::Single), m_single_val(value) {}
        IndexDesc(AllSlice) : m_type(IndexType::All) {}
        IndexDesc(Slice slice_value) : m_type(IndexType::Range), m_slice_val(std::move(slice_value)) {}
    };

    // METADATA
    struct RedMeta{
        std::vector<int64_t> temp_shape;
        std::vector<int64_t> temp_strides;
        std::vector<int64_t> result_shape;
        std::vector<int64_t> result_strides;
        int64_t result_vol;
        int64_t reduced_vol;
    };

    struct FusedView {
        std::vector<int64_t> shared_shape;
        std::vector<std::vector<int64_t>> strides;
    };

    enum MatrixTensorOp {
        Normal,
        Transposed
    };

    struct NMMMeta {
        std::vector<int64_t> result_shape;

        float alpha;
        float beta;

        int64_t M;
        int64_t K;
        int64_t N;
        
        MatrixTensorOp left_op;
        MatrixTensorOp right_op;

        int64_t lda;
        int64_t ldb;
        int64_t ldc;
    };

    struct BMMMeta {
        std::vector<int64_t> result_shape;

        float alpha;
        float beta;

        int64_t M;
        int64_t K;
        int64_t N;
        
        MatrixTensorOp left_op;
        MatrixTensorOp right_op;

        int64_t lda;
        int64_t ldb;
        int64_t ldc;

        int64_t stride_a;
        int64_t stride_b;
        int64_t stride_c;

        int64_t batch_count;
    };


    // SETTINGS
    struct PrintOptions {
        int64_t edge_items = 5;
        int dim_indentation = 2;
        bool show_metadata = true;
    };

    // TAGS
    struct LazyTag{};
    inline constexpr LazyTag lazy = LazyTag();

    struct UninitializedTag{};
    inline constexpr UninitializedTag uninitialized = UninitializedTag();
}