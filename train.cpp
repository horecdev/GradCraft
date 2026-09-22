#include "gradc/gradc.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>

using namespace gradc;

int main(int argc, char* argv[]) {
    try {
        // defaults
        std::string dataset_path = "./data/datasets/dataset.bin";
        std::string model_dir = "./models/mallmoc";
        int64_t total_steps = 6300;
        bool load_checkpoint = false;

        GPTConfig cfg;
        cfg.vocab_size = 8192;
        cfg.max_seq_len = 512;
        cfg.embed_dim = 768;
        cfg.num_heads = 12;
        cfg.num_layers = 11;

        int64_t B_target = 512; 
        int64_t B_real = 16;
        
        float max_lr = 6e-4f;
        float min_lr = 6e-5f;
        float beta1 = 0.9f;
        float beta2 = 0.95f;
        
        int64_t print_every = 10;
        int64_t checkpoint_every = 500;

        // argparse
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dataset" && i + 1 < argc) {dataset_path = argv[++i];}
            else if (arg == "--model_dir" && i + 1 < argc) {model_dir = argv[++i];}
            else if (arg == "--steps" && i + 1 < argc) {total_steps = std::stoll(argv[++i]);}
            else if (arg == "--vocab_size" && i + 1 < argc) {cfg.vocab_size = std::stoll(argv[++i]);}
            else if (arg == "--seq_len" && i + 1 < argc) {cfg.max_seq_len = std::stoll(argv[++i]);}
            else if (arg == "--embed_dim" && i + 1 < argc) {cfg.embed_dim = std::stoll(argv[++i]);}
            else if (arg == "--num_heads" && i + 1 < argc) {cfg.num_heads = std::stoll(argv[++i]);}
            else if (arg == "--num_layers" && i + 1 < argc) {cfg.num_layers = std::stoll(argv[++i]);}
            else if (arg == "--batch_size" && i + 1 < argc) {B_real = std::stoll(argv[++i]);}
            else if (arg == "--target_batch" && i + 1 < argc) {B_target = std::stoll(argv[++i]);}
            else if (arg == "--max_lr" && i + 1 < argc) {max_lr = std::stof(argv[++i]);}
            else if (arg == "--min_lr" && i + 1 < argc) {min_lr = std::stof(argv[++i]);}
            else if (arg == "--beta1" && i + 1 < argc) {beta1 = std::stof(argv[++i]);}
            else if (arg == "--beta2" && i + 1 < argc) {beta2 = std::stof(argv[++i]);}
            else if (arg == "--print_every" && i + 1 < argc) {print_every = std::stoll(argv[++i]);}
            else if (arg == "--checkpoint_every" && i + 1 < argc) {checkpoint_every = std::stoll(argv[++i]);}
            else if (arg == "--resume") {load_checkpoint = true;}
        }

        // derived hyperparams
        int64_t grad_accum_steps = B_target / B_real;
        int64_t warmup_steps = static_cast<int64_t>(total_steps * 0.05); // 5% is warmup


        if (!std::filesystem::exists(dataset_path)) {
            std::cerr << "Error: Dataset file not found at: " << dataset_path << "\nRun tokenizer first.\n";
            return 1;
        }

        std::filesystem::create_directories(model_dir);

        // model config, paths for saving, etc.
        cudaStream_t copy_stream = create_stream();
        cudaEvent_t event = create_event();
        Device gpu(DeviceType::CUDA, 0);
        Device cpu(DeviceType::CPU);

        float optim_eps = 1e-5f;
        float weight_decay = 0.1f;

        DataLoader loader(dataset_path, cfg.max_seq_len, 67);
        
        float base_std = 0.02f;
        float residual_std = 0.02f / std::sqrt(2.0f * cfg.num_layers);
        NormalInit<float> base_init(0.0f, base_std);
        NormalInit<float> residual_init(0.0f, residual_std);

        GPT<float> model(cfg.vocab_size, cfg.max_seq_len, cfg.embed_dim, cfg.num_heads, cfg.num_layers, base_init, residual_init, cfg.calc_eps);
        model.to(gpu);

        AdamW<float> optimizer(model.named_parameters(), 0.0f, beta1, beta2, weight_decay, optim_eps);
        CosineScheduler<float> scheduler(&optimizer, max_lr, min_lr, warmup_steps, total_steps);
        GlobalNormClipper<float> clipper(1.0f);

        std::string config_save_path = model_dir + "/config.bin";
        std::string latest_model_path = model_dir + "/latest_model.bin";
        std::string latest_optim_path = model_dir + "/latest_optim.bin";
        std::string latest_scheduler_path = model_dir + "/latest_scheduler.bin";
        std::string final_save_path = model_dir + "/trained_model.bin";
        std::string loss_log_path = model_dir + "/training_log.csv";

        save_gpt_config(cfg, config_save_path);

        // resuming
        int64_t start_step = 0;
        if (load_checkpoint) {
            if (!std::filesystem::exists(latest_model_path)) {
                std::cerr << "Error: Cannot resume. Checkpoint not found at: " << latest_model_path << std::endl;
                return 1;
            }
            std::cout << "Loading checkpoints from: " << model_dir << std::endl;
            
            auto model_state = load_tensor_checkpoint<float>(latest_model_path);
            model.load_state_dict(model_state);

            auto optim_state = load_tensor_checkpoint<float>(latest_optim_path);
            optimizer.load_state_dict(optim_state);

            auto scheduler_state = load_scalar_checkpoint<float>(latest_scheduler_path);
            scheduler.load_state_dict(scheduler_state);

            start_step = scheduler.m_t;
            std::cout << "Successfully resumed state from step: " << start_step << std::endl;
        }

        bool log_exists = std::filesystem::exists(loss_log_path);
        std::ofstream log_file(loss_log_path, std::ios::app);
        if (!log_file) {throw std::runtime_error("Failed to open log file.");}
        if (!log_exists || start_step == 0) {log_file << "step,loss,norm,lr,tok_per_sec\n";}


        // TRAINING LOOP!!!
        std::cout << "Starting training. Parameters: " << model.num_params() << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        float last_loss_val = 0.0f;
        int64_t tokens_per_interval = print_every * grad_accum_steps * B_real * cfg.max_seq_len;

        for (int64_t step = start_step; step < total_steps; ++step) {
            model.zero_grad();
            Tensor<float> step_loss_accum = Tensor<float>::zeros({}, gpu);

            for (int64_t micro_batch = 0; micro_batch < grad_accum_steps; ++micro_batch) {
                auto [X, Y] = loader.next_batch(B_real, cfg.max_seq_len, Device(DeviceType::CPU));
                X = X.to_async(gpu, copy_stream, event);
                Y = Y.to_async(gpu, copy_stream, event);
                
                Tensor<float> logits = model.forward(X);

                Tensor<float> loss = softmax_crossentropy_fast<float>(logits, Y, cfg.calc_eps);
                Tensor<float> scaled_loss = loss / static_cast<float>(grad_accum_steps);

                scaled_loss.realize();
                dispatch(gpu, BinaryOpInPlace::Add, step_loss_accum, scaled_loss);

                if (step % print_every == 0 && micro_batch == grad_accum_steps - 1) {
                    last_loss_val = step_loss_accum.item();
                }

                scaled_loss.backward();
            }
            float global_norm = clipper.normalize(model.parameters());

            scheduler.step();
            optimizer.step();

            if (step % print_every == 0) {
                auto end_time = std::chrono::high_resolution_clock::now();
                double interval_seconds = std::chrono::duration<double>(end_time - start_time).count();
                double tok_per_sec = tokens_per_interval / interval_seconds;
                
                std::cout << "STEP: " << step << " | LOSS: " << last_loss_val << " | NORM: " << global_norm << " | LR: " << scheduler.m_lr << " | TOK/S: " << tok_per_sec << std::endl;
                log_file << step << "," << last_loss_val << "," << global_norm << "," << scheduler.m_lr << "," << tok_per_sec << "\n";
                log_file.flush();
                start_time = std::chrono::high_resolution_clock::now();
            }

            if (step > start_step && step % checkpoint_every == 0) {
                std::cout << "Saving checkpoint at step: " << step <<  "at:" << latest_model_path << std::endl;
                save_tensor_checkpoint(model.state_dict(cpu), latest_model_path);
                save_tensor_checkpoint(optimizer.state_dict(cpu), latest_optim_path);
                save_scalar_checkpoint(scheduler.state_dict(), latest_scheduler_path);
            }
        }

        std::cout << "Saving final model to: " << final_save_path << std::endl;
        save_tensor_checkpoint(model.state_dict(cpu), final_save_path);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
}