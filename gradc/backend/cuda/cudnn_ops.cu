#include "gradc/backend/cuda/cuda_math.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <unordered_map>
#include <string>
#include <memory>

namespace gradc {
    template <typename T>
    constexpr cudnnDataType_t get_cudnn_dtype() {
        if constexpr (std::is_same_v<T, float>) return CUDNN_DATA_FLOAT;
        else if constexpr (std::is_same_v<T, double>) return CUDNN_DATA_DOUBLE;
        else throw std::runtime_error("Unsupported cuDNN data type.");
    }

    inline cudnnHandle_t get_cudnn_handle() {
        static cudnnHandle_t handle = nullptr;
        if (handle == nullptr) {
            if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
                throw std::runtime_error("cuDNN handle initialization failed.");
            }
        }

        return handle;
    }

    #pragma region CUDNN LAYERNORM

    namespace fe = cudnn_frontend;

    // cache structs
    struct LNFwdCache {
        std::shared_ptr<fe::graph::Graph> graph;
        fe::graph::Tensor_attributes::uid_t x_uid, gamma_uid, beta_uid, y_uid, mean_uid, inv_var_uid;
        int64_t workspace_size;
    };

    struct LNBwdCache {
        std::shared_ptr<fe::graph::Graph> graph;
        fe::graph::Tensor_attributes::uid_t dy_uid, x_uid, gamma_uid, mean_uid, inv_var_uid;
        fe::graph::Tensor_attributes::uid_t dx_uid, dgamma_uid, dbeta_uid;
        int64_t workspace_size;
    };

    template <typename T>
    constexpr fe::DataType_t get_cudnn_fe_dtype() {
        if constexpr (std::is_same_v<T, float>) {
            return fe::DataType_t::FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            return fe::DataType_t::DOUBLE;
        }
    }

    template <typename T>
    requires std::is_floating_point_v<T>
    void CUDAMath::apply_cudnn_layernorm_forward(Tensor<T>& out, Tensor<T>& saved_mean, Tensor<T>& saved_inv_var, const Tensor<T>& x, const Tensor<T>& gamma, const Tensor<T>& beta, T eps, bool save_intermediates) {
        cudnnHandle_t handle = get_cudnn_handle();
        cudaSetDevice(x.device().index);

        // get ptrs
        void* p_x     = static_cast<void*>(x._get_storage()->data() + x.offset());
        void* p_y     = static_cast<void*>(out._get_storage()->data() + out.offset());
        void* p_gamma = static_cast<void*>(gamma._get_storage()->data() + gamma.offset());
        void* p_beta  = static_cast<void*>(beta._get_storage()->data() + beta.offset());
        
        void* p_mean    = nullptr;
        void* p_inv_var = nullptr;
        if (save_intermediates) {
            p_mean    = static_cast<void*>(saved_mean._get_storage()->data() + saved_mean.offset());
            p_inv_var = static_cast<void*>(saved_inv_var._get_storage()->data() + saved_inv_var.offset());
        }

        fe::DataType_t dtype = get_cudnn_fe_dtype<T>();

        // generate unique key based on shape and training/inference
        std::string shape_key = "";
        for (int64_t d : x.shape()) shape_key += std::to_string(d) + "_";
        shape_key += (save_intermediates ? "TRAIN" : "INFER");

        // cache map
        static std::unordered_map<std::string, LNFwdCache> fwd_cache;

        // compile only if cache not in the map
        if (fwd_cache.find(shape_key) == fwd_cache.end()) {
            auto graph = std::make_shared<fe::graph::Graph>();
            
            auto x_tensor = graph->tensor(
                fe::graph::Tensor_attributes().set_name("X").set_data_type(dtype).set_dim(x.shape()).set_stride(x.strides())
            );

            auto gamma_tensor = graph->tensor(
                fe::graph::Tensor_attributes().set_name("Gamma").set_data_type(dtype).set_dim(gamma.shape()).set_stride(gamma.strides())
            );

            auto beta_tensor = graph->tensor(
                fe::graph::Tensor_attributes().set_name("Beta").set_data_type(dtype).set_dim(beta.shape()).set_stride(beta.strides())
            );

            fe::NormFwdPhase_t phase = save_intermediates ? fe::NormFwdPhase_t::TRAINING : fe::NormFwdPhase_t::INFERENCE;
            auto layernorm_attr = fe::graph::Layernorm_attributes().set_forward_phase(phase).set_epsilon(static_cast<double>(eps));

            auto [y_tensor, mean_tensor, inv_var_tensor] = graph->layernorm(x_tensor, gamma_tensor, beta_tensor, layernorm_attr);

            y_tensor->set_output(true).set_data_type(dtype).set_dim(out.shape()).set_stride(out.strides());

            if (save_intermediates) {
                mean_tensor->set_output(true).set_data_type(dtype).set_dim(saved_mean.shape()).set_stride(saved_mean.strides());
                inv_var_tensor->set_output(true).set_data_type(dtype).set_dim(saved_inv_var.shape()).set_stride(saved_inv_var.strides());
            }

            if (!graph->validate().is_good() || !graph->build_operation_graph(handle).is_good() || 
                !graph->create_execution_plans({fe::HeurMode_t::A}).is_good() || !graph->check_support().is_good() || 
                !graph->build_plans().is_good()) {
                throw std::runtime_error("cuDNN FWD Graph compilation failed.");
            }

            LNFwdCache entry;
            entry.graph = graph;
            entry.x_uid = x_tensor->get_uid();
            entry.gamma_uid = gamma_tensor->get_uid();
            entry.beta_uid = beta_tensor->get_uid();
            entry.y_uid = y_tensor->get_uid();
            if (save_intermediates) {
                entry.mean_uid = mean_tensor->get_uid();
                entry.inv_var_uid = inv_var_tensor->get_uid();
            }
            auto smth = graph->get_workspace_size(entry.workspace_size);
            fwd_cache[shape_key] = entry;
        }

        // retrieve from cache
        auto& cache_entry = fwd_cache[shape_key];
        void* workspace_ptr = nullptr;
        if (cache_entry.workspace_size > 0) {
            workspace_ptr = CUDAMemPool::get().allocate(cache_entry.workspace_size, x.device());
        }

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
            {cache_entry.x_uid, p_x},
            {cache_entry.gamma_uid, p_gamma},
            {cache_entry.beta_uid, p_beta},
            {cache_entry.y_uid, p_y}
        };

        if (save_intermediates) {
            variant_pack[cache_entry.mean_uid] = p_mean;
            variant_pack[cache_entry.inv_var_uid] = p_inv_var;
        }

        // execute the math
        auto status = cache_entry.graph->execute(handle, variant_pack, workspace_ptr);

        // free CUDA memory back into mempool
        if (cache_entry.workspace_size > 0) {
            CUDAMemPool::get().free(workspace_ptr, cache_entry.workspace_size, x.device());
        }

        if (!status.is_good()) {
            throw std::runtime_error("cuDNN LayerNorm FWD graph execution failed: " + status.get_message());
        }
    }

    template <typename T>
    requires std::is_floating_point_v<T>
    void CUDAMath::apply_cudnn_layernorm_backward(Tensor<T>& dx, Tensor<T>& dgamma, Tensor<T>& dbeta, const Tensor<T>& dy, const Tensor<T>& x, const Tensor<T>& gamma, const Tensor<T>& saved_mean, const Tensor<T>& saved_inv_var) {
        cudnnHandle_t handle = get_cudnn_handle();
        cudaSetDevice(x.device().index);

        void* p_dx = static_cast<void*>(dx._get_storage()->data() + dx.offset());
        void* p_dgamma = static_cast<void*>(dgamma._get_storage()->data() + dgamma.offset());
        void* p_dbeta = static_cast<void*>(dbeta._get_storage()->data() + dbeta.offset());
        void* p_dy = static_cast<void*>(dy._get_storage()->data() + dy.offset());
        void* p_x = static_cast<void*>(x._get_storage()->data() + x.offset());
        void* p_gamma = static_cast<void*>(gamma._get_storage()->data() + gamma.offset());
        void* p_mean = static_cast<void*>(saved_mean._get_storage()->data() + saved_mean.offset());
        void* p_inv_var = static_cast<void*>(saved_inv_var._get_storage()->data() + saved_inv_var.offset());

        fe::DataType_t dtype = get_cudnn_fe_dtype<T>();

        // 1. Generate unique string key based on shape
        std::string shape_key = "";
        for (int64_t d : x.shape()) shape_key += std::to_string(d) + "_";

        // 2. Static cache map
        static std::unordered_map<std::string, LNBwdCache> bwd_cache;

        // 3. Compile graph ONLY if not found in cache
        if (bwd_cache.find(shape_key) == bwd_cache.end()) {
            auto graph = std::make_shared<fe::graph::Graph>();
            
            auto dy_tensor = graph->tensor(fe::graph::Tensor_attributes().set_name("dY").set_data_type(dtype).set_dim(dy.shape()).set_stride(dy.strides()));
            auto x_tensor = graph->tensor(fe::graph::Tensor_attributes().set_name("X").set_data_type(dtype).set_dim(x.shape()).set_stride(x.strides()));
            auto gamma_tensor = graph->tensor(fe::graph::Tensor_attributes().set_name("Gamma").set_data_type(dtype).set_dim(gamma.shape()).set_stride(gamma.strides()));
            auto mean_tensor = graph->tensor(fe::graph::Tensor_attributes().set_name("Mean").set_data_type(dtype).set_dim(saved_mean.shape()).set_stride(saved_mean.strides()));
            auto inv_var_tensor = graph->tensor(fe::graph::Tensor_attributes().set_name("InvVar").set_data_type(dtype).set_dim(saved_inv_var.shape()).set_stride(saved_inv_var.strides()));

            auto bwd_attr = fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(mean_tensor, inv_var_tensor);

            auto [dx_tensor, dgamma_tensor, dbeta_tensor] = graph->layernorm_backward(dy_tensor, x_tensor, gamma_tensor, bwd_attr);

            dx_tensor->set_output(true).set_data_type(dtype).set_dim(dx.shape()).set_stride(dx.strides());
            dgamma_tensor->set_output(true).set_data_type(dtype).set_dim(dgamma.shape()).set_stride(dgamma.strides());
            dbeta_tensor->set_output(true).set_data_type(dtype).set_dim(dbeta.shape()).set_stride(dbeta.strides());

            if (!graph->validate().is_good() || !graph->build_operation_graph(handle).is_good() || 
                !graph->create_execution_plans({fe::HeurMode_t::A}).is_good() || !graph->check_support().is_good() || 
                !graph->build_plans().is_good()) {
                throw std::runtime_error("cuDNN BWD Graph compilation failed.");
            }

            LNBwdCache entry;
            entry.graph = graph;
            entry.dy_uid = dy_tensor->get_uid();
            entry.x_uid = x_tensor->get_uid();
            entry.gamma_uid = gamma_tensor->get_uid();
            entry.mean_uid = mean_tensor->get_uid();
            entry.inv_var_uid = inv_var_tensor->get_uid();
            entry.dx_uid = dx_tensor->get_uid();
            entry.dgamma_uid = dgamma_tensor->get_uid();
            entry.dbeta_uid = dbeta_tensor->get_uid();
            auto smth = graph->get_workspace_size(entry.workspace_size);
            bwd_cache[shape_key] = entry;
        }

        // 4. Retrieve from cache and bind active memory pointers
        auto& cache_entry = bwd_cache[shape_key];
        void* workspace_ptr = nullptr;
        if (cache_entry.workspace_size > 0) {
            workspace_ptr = CUDAMemPool::get().allocate(cache_entry.workspace_size, x.device());
        }

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
            {cache_entry.dy_uid, p_dy},
            {cache_entry.x_uid, p_x},
            {cache_entry.gamma_uid, p_gamma},
            {cache_entry.mean_uid, p_mean},
            {cache_entry.inv_var_uid, p_inv_var},
            {cache_entry.dx_uid, p_dx},
            {cache_entry.dgamma_uid, p_dgamma},
            {cache_entry.dbeta_uid, p_dbeta}
        };
        
        auto status = cache_entry.graph->execute(handle, variant_pack, workspace_ptr);

        if (cache_entry.workspace_size > 0) {
            CUDAMemPool::get().free(workspace_ptr, cache_entry.workspace_size, x.device());
        }

        if (!status.is_good()) {
            throw std::runtime_error("cuDNN LayerNorm backward graph execution failed: " + status.get_message());
        }
    }

    #pragma endregion CUDNN LAYERNORM

    #pragma region TEMPLATING

    #define INSTANTIATE_CUDNN_LAYERNORM(T) \
        template void CUDAMath::apply_cudnn_layernorm_forward<T>(Tensor<T>&, Tensor<T>&, Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&, T, bool); \
        template void CUDAMath::apply_cudnn_layernorm_backward<T>(Tensor<T>&, Tensor<T>&, Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&);

    INSTANTIATE_CUDNN_LAYERNORM(float)
    INSTANTIATE_CUDNN_LAYERNORM(double)

    #pragma endregion TEMPLATING
}