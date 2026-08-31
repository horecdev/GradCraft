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

    #pragma region CUDNN SDPA

    template <typename T>
    requires std::is_same_v<T, float>
    void CUDAMath::apply_cudnn_sdpa_forward(Tensor<T>& out, Tensor<T>& saved_lse, const Tensor<T>& q, const Tensor<T>& k, const Tensor<T>& v, T scale, bool is_causal, bool save_intermediates) {
        cudnnHandle_t handle = get_cudnn_handle();
        cudaSetDevice(q.device().index);

        void* p_q   = static_cast<void*>(q._get_storage()->data() + q.offset());
        void* p_k   = static_cast<void*>(k._get_storage()->data() + k.offset());
        void* p_v   = static_cast<void*>(v._get_storage()->data() + v.offset());
        void* p_out = static_cast<void*>(out._get_storage()->data() + out.offset());
        
        void* p_lse = nullptr;
        if (save_intermediates) {
            p_lse = static_cast<void*>(saved_lse._get_storage()->data() + saved_lse.offset());
        }

        fe::DataType_t dtype = fe::DataType_t::FLOAT;
        fe::graph::Graph graph;

        // input tensors
        auto q_tensor = graph.tensor(
            fe::graph::Tensor_attributes().set_name("Q").set_data_type(dtype)
            .set_dim(q.shape()).set_stride(q.strides())
        );
        auto k_tensor = graph.tensor(
            fe::graph::Tensor_attributes().set_name("K").set_data_type(dtype)
            .set_dim(k.shape()).set_stride(k.strides())
        );
        auto v_tensor = graph.tensor(
            fe::graph::Tensor_attributes().set_name("V").set_data_type(dtype)
            .set_dim(v.shape()).set_stride(v.strides())
        );

        auto sdpa_attr = fe::graph::SDPA_attributes()
            .set_generate_stats(save_intermediates)
            .set_attn_scale(static_cast<float>(scale));

        if (is_causal) {
            // ignore everything on the top left (causal)
            sdpa_attr.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT);
            sdpa_attr.set_diagonal_band_right_bound(0);
        }

        // connect inputs and create outputs
        auto [out_tensor, lse_tensor] = graph.sdpa(q_tensor, k_tensor, v_tensor, sdpa_attr);

        // configure outputs
        out_tensor->set_output(true)
            .set_data_type(dtype)
            .set_dim(out.shape())
            .set_stride(out.strides());

        if (save_intermediates) {
            // only floats (function restricted for floats also)
            lse_tensor->set_output(true)
                .set_data_type(fe::DataType_t::FLOAT) 
                .set_dim(saved_lse.shape())
                .set_stride(saved_lse.strides());
        }

        // compile graph
        auto status = graph.validate();
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA validation failed: " + status.get_message());

        status = graph.build_operation_graph(handle);
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA op graph failed: " + status.get_message());

        status = graph.create_execution_plans({fe::HeurMode_t::A});
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA exec plans failed: " + status.get_message());

        status = graph.check_support();
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA check support failed: " + status.get_message());

        status = graph.build_plans();
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA build plans failed: " + status.get_message());

        // workspace memory
        int64_t workspace_size = 0;
        auto smth = graph.get_workspace_size(workspace_size);
        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace_ptr = CUDAMemPool::get().allocate(workspace_size, q.device());
        }

        // bind pointers
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
            {q_tensor->get_uid(), p_q},
            {k_tensor->get_uid(), p_k},
            {v_tensor->get_uid(), p_v},
            {out_tensor->get_uid(), p_out}
        };

        if (save_intermediates) {
            variant_pack[lse_tensor->get_uid()] = p_lse;
        }

        status = graph.execute(handle, variant_pack, workspace_ptr);

        // free memory
        if (workspace_size > 0) {
            CUDAMemPool::get().free(workspace_ptr, workspace_size, q.device());
        }

        if (!status.is_good()) {
            throw std::runtime_error("cuDNN SDPA execution failed: " + status.get_message());
        }
    }

    template <typename T>
    requires std::is_same_v<T, float>
    void CUDAMath::apply_cudnn_sdpa_backward(Tensor<T>& dq, Tensor<T>& dk, Tensor<T>& dv, const Tensor<T>& out_grad, const Tensor<T>& q, const Tensor<T>& k, const Tensor<T>& v, const Tensor<T>& out, const Tensor<T>& saved_lse, T scale, bool is_causal) {
        cudnnHandle_t handle = get_cudnn_handle();
        cudaSetDevice(q.device().index);

        void* p_dq = static_cast<void*>(dq._get_storage()->data() + dq.offset());
        void* p_dk = static_cast<void*>(dk._get_storage()->data() + dk.offset());
        void* p_dv = static_cast<void*>(dv._get_storage()->data() + dv.offset());
        void* p_q = static_cast<void*>(q._get_storage()->data() + q.offset());
        void* p_k = static_cast<void*>(k._get_storage()->data() + k.offset());
        void* p_v = static_cast<void*>(v._get_storage()->data() + v.offset());
        void* p_out = static_cast<void*>(out._get_storage()->data() + out.offset());
        void* p_out_grad = static_cast<void*>(out_grad._get_storage()->data() + out_grad.offset());
        void* p_stats = static_cast<void*>(saved_lse._get_storage()->data() + saved_lse.offset());

        fe::DataType_t dtype = fe::DataType_t::FLOAT;
        fe::graph::Graph graph;

        auto q_tensor = graph.tensor(fe::graph::Tensor_attributes().set_name("Q").set_data_type(dtype).set_dim(q.shape()).set_stride(q.strides()));
        auto k_tensor = graph.tensor(fe::graph::Tensor_attributes().set_name("K").set_data_type(dtype).set_dim(k.shape()).set_stride(k.strides()));
        auto v_tensor = graph.tensor(fe::graph::Tensor_attributes().set_name("V").set_data_type(dtype).set_dim(v.shape()).set_stride(v.strides()));
        auto out_tensor = graph.tensor(fe::graph::Tensor_attributes().set_name("O").set_data_type(dtype).set_dim(out.shape()).set_stride(out.strides()));
        auto dY_tensor = graph.tensor(fe::graph::Tensor_attributes().set_name("dY").set_data_type(dtype).set_dim(out_grad.shape()).set_stride(out_grad.strides()));
        
        auto stats_tensor = graph.tensor(fe::graph::Tensor_attributes().set_name("Stats").set_data_type(fe::DataType_t::FLOAT).set_dim(saved_lse.shape()).set_stride(saved_lse.strides()));

        auto sdpa_bwd_attr = fe::graph::SDPA_backward_attributes()
            .set_attn_scale(scale);

        if (is_causal) {
            sdpa_bwd_attr.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT);
            sdpa_bwd_attr.set_diagonal_band_right_bound(0);
        }

        auto [dq_tensor, dk_tensor, dv_tensor] = graph.sdpa_backward(
            q_tensor, k_tensor, v_tensor, out_tensor, dY_tensor, stats_tensor, sdpa_bwd_attr
        );

        dq_tensor->set_output(true).set_data_type(dtype).set_dim(dq.shape()).set_stride(dq.strides());
        dk_tensor->set_output(true).set_data_type(dtype).set_dim(dk.shape()).set_stride(dk.strides());
        dv_tensor->set_output(true).set_data_type(dtype).set_dim(dv.shape()).set_stride(dv.strides());

        auto status = graph.validate();
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA BWD validation failed: " + status.get_message());

        status = graph.build_operation_graph(handle);
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA BWD op graph failed: " + status.get_message());

        status = graph.create_execution_plans({fe::HeurMode_t::A});
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA BWD exec plans failed: " + status.get_message());

        status = graph.check_support();
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA BWD check support failed: " + status.get_message());

        status = graph.build_plans();
        if (!status.is_good()) throw std::runtime_error("cuDNN SDPA BWD build plans failed: " + status.get_message());

        int64_t workspace_size = 0;
        auto smth = graph.get_workspace_size(workspace_size);
        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            workspace_ptr = CUDAMemPool::get().allocate(workspace_size, q.device());
        }

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
            {q_tensor->get_uid(), p_q},
            {k_tensor->get_uid(), p_k},
            {v_tensor->get_uid(), p_v},
            {out_tensor->get_uid(), p_out},
            {dY_tensor->get_uid(), p_out_grad},
            {stats_tensor->get_uid(), p_stats},
            {dq_tensor->get_uid(), p_dq},
            {dk_tensor->get_uid(), p_dk},
            {dv_tensor->get_uid(), p_dv}
        };

        status = graph.execute(handle, variant_pack, workspace_ptr);

        if (workspace_size > 0) {
            CUDAMemPool::get().free(workspace_ptr, workspace_size, q.device());
        }

        if (!status.is_good()) {
            throw std::runtime_error("cuDNN SDPA BWD execution failed: " + status.get_message());
        }
    }

    #pragma endregion CUDNN SDPA

    #pragma region TEMPLATING

    #define INSTANTIATE_CUDNN_LAYERNORM(T) \
        template void CUDAMath::apply_cudnn_layernorm_forward<T>(Tensor<T>&, Tensor<T>&, Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&, T, bool); \
        template void CUDAMath::apply_cudnn_layernorm_backward<T>(Tensor<T>&, Tensor<T>&, Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&, const Tensor<T>&);

    INSTANTIATE_CUDNN_LAYERNORM(float)
    INSTANTIATE_CUDNN_LAYERNORM(double)


    #pragma endregion TEMPLATING
}