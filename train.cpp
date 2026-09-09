#include "gradc/gradc.hpp"
#include <iostream>
#include <chrono>

using namespace gradc;
int main() {
    try {
        // CONFIG
        cudaStream_t copy_stream = create_stream();
        cudaEvent_t event = create_event();
        Device gpu(DeviceType::CUDA, 0);
        Device cpu(DeviceType::CPU);

        // HYPERPARAMS
        int64_t B_target = 510; // 85 * 6 = 510
        int64_t B_real = 6; // target 6 for 3090
        int64_t seq_len = 1024;
        int64_t vocab_size = 32768;
        int64_t embed_dim = 768;
        int64_t num_heads = 12;
        int64_t num_layers = 16;

        // OPTIMIZER / SCHEDULER HYPERPARAMS
        float max_lr = 3e-4f;
        float min_lr = 3e-5f;
        float calc_eps = 1e-5f;
        float optim_eps = 1e-8f;

        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float weight_decay = 0.1f;

        // TRAINING HYPERPARAMS
        int64_t grad_accum_steps = B_target / B_real;
        int64_t total_steps = 8'272; // 8272 * 510 * 1024 = 4.3 billion tokens
        int64_t warmup_steps = 0; // 10%

        // DATA
        DataLoader loader = DataLoader("C:/Local Projects/autograd_cpp/data/datasets/cosmo_cpp.bin");
        
        // MODEL
        float std = 1 / std::sqrt(2 * num_layers);
        NormalInit<float> init(0.0f, std);
        GPT<float> model(vocab_size, seq_len, embed_dim, num_heads, num_layers, init, calc_eps);
        model.to(gpu);

        // OPTIMIZER AND SCHEDULER
        AdamW<float> optimizer(model.named_parameters(), 0.0f, beta1, beta2, weight_decay, optim_eps);
        CosineScheduler<float> scheduler(&optimizer, max_lr, min_lr, warmup_steps, total_steps);

        // CHECKPOINTING
        bool load_checkpoint = true;
        int64_t checkpoint_every = 500;

        std::string latest_model_path = "C:/Local Projects/autograd_cpp/models/MALLMOC-174/latest_model.bin";
        std::string latest_optim_path = "C:/Local Projects/autograd_cpp/models/MALLMOC-174/latest_optim.bin";
        std::string latest_scheduler_path = "C:/Local Projects/autograd_cpp/models/MALLMOC-174/latest_scheduler.bin";

        std::string final_save_path = "C:/Local Projects/autograd_cpp/models/MALLMOC-174/trained_model.bin";

        int64_t start_step = 0;
        if (load_checkpoint != false) {
            std::cout << "Loading checkpoint..." << std::endl;

            auto model_state = load_tensor_checkpoint<float>(latest_model_path);
            model.load_state_dict(model_state);

            auto optim_state = load_tensor_checkpoint<float>(latest_optim_path);
            optimizer.load_state_dict(optim_state);

            auto scheduler_state = load_scalar_checkpoint<float>(latest_scheduler_path);
            scheduler.load_state_dict(scheduler_state);

            start_step = scheduler.m_t;
            std::cout << "Successfully loaded state from step: " << start_step << std::endl;
        }

        // LOG
        int64_t print_every = 2;
        int64_t tokens_per_interval = print_every * grad_accum_steps * B_real * seq_len;
        std::string loss_log_path = "C:/Local Projects/autograd_cpp/models/MALLMOC-174/training_log.csv";
        bool log_exists = std::filesystem::exists(loss_log_path);
        std::ofstream log_file(loss_log_path, std::ios::app);
        if (!log_file) {
            throw std::runtime_error("Failed to open training log file.");
        }
        if (!log_exists || start_step == 0) {
            log_file << "step,loss,lr,tok_per_sec\n";
        }

        std::string num_params = std::format(std::locale("en_US.UTF-8"), "{:L}", model.num_params());
        std::cout << "Starting training of MALLMOC-174. Number of params: " << num_params << std::endl;;

        auto start_time = std::chrono::high_resolution_clock::now();
        float last_loss_val = 0.0f;

        for (int64_t step = start_step; step < total_steps; ++step) {
            model.zero_grad();
            for (int64_t micro_batch = 0; micro_batch < grad_accum_steps; ++micro_batch) {
                auto [X, Y] = loader.next_batch(B_real, seq_len, Device(DeviceType::CPU));
                X = X.to_async(gpu, copy_stream, event);
                Y = Y.to_async(gpu, copy_stream, event);
                Tensor<float> logits = model.forward(X);

                Tensor<float> loss = softmax_crossentropy_fast<float>(logits, Y, calc_eps);
                Tensor<float> scaled_loss = loss / static_cast<float>(grad_accum_steps); // SCEL does 1/6 but u gotta do 1/510

                scaled_loss.realize();
                
                if (step % print_every == 0 && micro_batch == grad_accum_steps - 1) {
                    last_loss_val = loss.item();
                }

                scaled_loss.backward();
            }
            scheduler.step();
            optimizer.step();

            if (step % print_every == 0) {
                auto end_time = std::chrono::high_resolution_clock::now();
                double interval_seconds = std::chrono::duration<double>(end_time - start_time).count();
                double tok_per_sec = tokens_per_interval / interval_seconds;
                
                std::cout << "LOG| step: " << step << " | loss: " << last_loss_val << " | lr: " << scheduler.m_lr << " | tok/s: " << tok_per_sec << std::endl;
                log_file << step << "," << last_loss_val << "," << scheduler.m_lr << "," << tok_per_sec << "\n";
                log_file.flush(); // force to write
                          
                start_time = std::chrono::high_resolution_clock::now(); // reset the timer
            }

            if (step > start_step && step % checkpoint_every == 0) {
                std::cout << "Saving checkpoint at step: " << step << std::endl;

                save_tensor_checkpoint(model.state_dict(cpu), latest_model_path);
                save_tensor_checkpoint(optimizer.state_dict(cpu), latest_optim_path);
                save_scalar_checkpoint(scheduler.state_dict(), latest_scheduler_path);

                std::cout << "Checkpoint finished successfully." << std::endl;
            }
        }

        std::cout << "Training of MALLMOC-174 finished. Saving final model to: " << final_save_path;

        save_tensor_checkpoint(model.state_dict(cpu), final_save_path);
        
        std::cout << "Saving successful." << std::endl;


        return 0;
        
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}