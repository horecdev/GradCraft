#include "gradc/backend/cuda/cuda_math.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

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

    template <typename T>
    constexpr fe::DataType_t get_cudnn_fe_dtype() {
        if constexpr (std::is_same_v<T, float>) {
            return fe::DataType_t::FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            return fe::DataType_t::DOUBLE;
        } else if constexpr (std::is_same_v<T, uint16_t>) { // e.g., fp16 / __half cast
            return fe::DataType_t::HALF;
        } else {
            return fe::DataType_t::FLOAT;
        }
    }

    template <typename T>
    requires std::is_floating_point_v<T>
    void CUDAMath::apply_cudnn_layernorm_forward(Tensor<T>& out, Tensor<T>& saved_mean, Tensor<T>& saved_inv_var, const Tensor<T>& x, const Tensor<T>& gamma, const Tensor<T>& beta, T eps, bool save_intermediates) {
        cudnnHandle_t handle = get_cudnn_handle();
        cudaSetDevice(x.device().index);

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

        fe::graph::Graph graph;

        // literally create tensor variables in cuDNN with these lines
        auto x_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("X")
                .set_data_type(dtype)
                .set_dim(x.shape())
                .set_stride(x.strides())
        );

        auto gamma_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("Gamma")
                .set_data_type(dtype)
                .set_dim(gamma.shape())
                .set_stride(gamma.strides())
        );

        auto beta_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("Beta")
                .set_data_type(dtype)
                .set_dim(beta.shape())
                .set_stride(beta.strides())
        );

        // do we need to save inv_var or mean?
        fe::NormFwdPhase_t phase = save_intermediates ? fe::NormFwdPhase_t::TRAINING : fe::NormFwdPhase_t::INFERENCE;
        // configure layernorm
        auto layernorm_attr = fe::graph::Layernorm_attributes()
            .set_forward_phase(phase)
            .set_epsilon(static_cast<double>(eps));

        // this line below is what connects outputs to inputs via layernorm. Creates 3 new tensors (outputs)
        auto [y_tensor, mean_tensor, inv_var_tensor] = graph.layernorm(x_tensor, gamma_tensor, beta_tensor, layernorm_attr);

        // after they are already plugged
        y_tensor->set_output(true)
                .set_data_type(dtype)
                .set_dim(out.shape())
                .set_stride(out.strides());

        if (save_intermediates) {
            mean_tensor->set_output(true)
                    .set_data_type(dtype)
                    .set_dim(saved_mean.shape())
                    .set_stride(saved_mean.strides());

            inv_var_tensor->set_output(true)
                        .set_data_type(dtype)
                        .set_dim(saved_inv_var.shape())
                        .set_stride(saved_inv_var.strides());
        }

        auto status = graph.validate();
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN graph validation failed: " + status.get_message());
        }

        status = graph.build_operation_graph(handle);
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN build_operation_graph failed: " + status.get_message());
        }

        status = graph.create_execution_plans({fe::HeurMode_t::A});
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN create_execution_plans failed: " + status.get_message());
        }

        status = graph.check_support();
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN check_support failed: " + status.get_message());
        }

        status = graph.build_plans();
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN build_plans failed: " + status.get_message());
        }


        // workspace - intermediate memory allocates using our custom CUDAMemPool
        int64_t workspace_size = 0;
        auto smth = graph.get_workspace_size(workspace_size);
        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace_ptr = CUDAMemPool::get().allocate(workspace_size, x.device());
        }

        // hook up actual memory pointers to tensors (both in and out)
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
            {x_tensor->get_uid(), p_x},
            {gamma_tensor->get_uid(), p_gamma},
            {beta_tensor->get_uid(), p_beta},
            {y_tensor->get_uid(), p_y}
        };

        if (save_intermediates) {
            variant_pack[mean_tensor->get_uid()] = p_mean;
            variant_pack[inv_var_tensor->get_uid()] = p_inv_var;
        }

        // execute the math
        status = graph.execute(handle, variant_pack, workspace_ptr);

        // free CUDA memory back into mempool
        if (workspace_size > 0) {
            CUDAMemPool::get().free(workspace_ptr, workspace_size, x.device());
        }

        if (!status.is_good()) {
            throw std::runtime_error("cuDNN LayerNorm graph execution failed: " + status.get_message());
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

        fe::graph::Graph graph;

        // create variable tensors (input)
        auto dy_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("dY")
                .set_data_type(dtype)
                .set_dim(dy.shape())
                .set_stride(dy.strides())
        );

        auto x_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("X")
                .set_data_type(dtype)
                .set_dim(x.shape())
                .set_stride(x.strides())
        );

        auto gamma_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("Gamma")
                .set_data_type(dtype)
                .set_dim(gamma.shape())
                .set_stride(gamma.strides())
        );

        auto mean_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("Mean")
                .set_data_type(dtype)
                .set_dim(saved_mean.shape())
                .set_stride(saved_mean.strides())
        );

        auto inv_var_tensor = graph.tensor(
            fe::graph::Tensor_attributes()
                .set_name("InvVar")
                .set_data_type(dtype)
                .set_dim(saved_inv_var.shape())
                .set_stride(saved_inv_var.strides())
        );

        auto bwd_attr = fe::graph::Layernorm_backward_attributes()
            .set_saved_mean_and_inv_variance(mean_tensor, inv_var_tensor);

        // create out tensors (derivative) and link them up
        auto [dx_tensor, dgamma_tensor, dbeta_tensor] = graph.layernorm_backward(
            dy_tensor, x_tensor, gamma_tensor, bwd_attr
        );

        // configure outputs
        dx_tensor->set_output(true)
                .set_data_type(dtype)
                .set_dim(dx.shape())
                .set_stride(dx.strides());

        dgamma_tensor->set_output(true)
                .set_data_type(dtype)
                .set_dim(dgamma.shape())
                .set_stride(dgamma.strides());

        dbeta_tensor->set_output(true)
                .set_data_type(dtype)
                .set_dim(dbeta.shape())
                .set_stride(dbeta.strides());

        auto status = graph.validate();
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN graph validation failed: " + status.get_message());
        }

        status = graph.build_operation_graph(handle);
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN build_operation_graph failed: " + status.get_message());
        }

        status = graph.create_execution_plans({fe::HeurMode_t::A});
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN create_execution_plans failed: " + status.get_message());
        }

        status = graph.check_support();
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN check_support failed: " + status.get_message());
        }

        status = graph.build_plans();
        if (!status.is_good()) {
            throw std::runtime_error("cuDNN build_plans failed: " + status.get_message());
        }

        // working memory
        int64_t workspace_size = 0;
        auto smth = graph.get_workspace_size(workspace_size);
        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace_ptr = CUDAMemPool::get().allocate(workspace_size, x.device());
        }

        // connect pointers
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
            {dy_tensor->get_uid(), p_dy},
            {x_tensor->get_uid(), p_x},
            {gamma_tensor->get_uid(), p_gamma},
            {mean_tensor->get_uid(), p_mean},
            {inv_var_tensor->get_uid(), p_inv_var},
            {dx_tensor->get_uid(), p_dx},
            {dgamma_tensor->get_uid(), p_dgamma},
            {dbeta_tensor->get_uid(), p_dbeta}
        };
        
        status = graph.execute(handle, variant_pack, workspace_ptr);

        if (workspace_size > 0) {
            CUDAMemPool::get().free(workspace_ptr, workspace_size, x.device());
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