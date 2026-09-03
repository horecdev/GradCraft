DONE Toposort
DONE 10 Backward passes
DONE Rename to view and alloc
DONE ConcatNode + bwd pass
DONE Slicing with ranges
DONE Implement dependencies to every node.
DONE Wipe grad memory after its no longer needed (current passed its grads down) (toposort)
DONE Write zero_grad to tensor class
DONE InPlaceOpps as disguise to normal opps (no nodes, just operator overload)
DONE Delete nodes on Inplace
DONE test out the bwd pass.
DONE InPlace mutation on REALZIE TIME if refcount tensorstate == 1 and refcount storage == 1. Its guaranteed to be an rvalue living only inside AddNode. Add to ALL applicable
DONE Check if requires grad to do calculation inside node backward() function
DONE Remove m_version from existence
DONE Unary nodes (ReLU) + framework
DONE Refactor storage to be 32-bit aligned for AVX, and hold an ENUM CPU/CUDA
DONE Update Tensor, TensorState constructors to support eager size/device propagation
DONE Reflect changes inside lazy and eager operations to use the new constructors and pass correct parameters
DONE Add assertions that devices are identical
DONE Fix the fill / write inefficiency (1GB writes for 500MB tensor for stuff with alloc) for CPU
DONE apply_unary_in_place + make contiguous use it
DONE BIG REFACTOR (hide backend from frontend)
DONE Update the out of, in, reduce etc. to write to uninitialized memory. think of the issue where memory is initialized and scalars.
DONE Write 6 dispatchers (out of, in, reduce, unary in, unary out of, layout) - everything that copies/creates new data gets a dispatcher to cuda/cpu
DONE Split into lob_alloc and lob_view files the lobotomized funcs so you can include views inside cpu_apply (they dont need dispatchers)
DONE Fix #includes
DONE Compile this big ass codebase
DONE SIMD contiguous fast paths (apply_in_place, apply_out_of_place, apply_unary)
DONE Memory pools for CPU (same shape math ran thousands of times)
DONE Download CUDA toolkit and set up the compiler + cmake
DONE Write memory allocations for CUDA (CUDAMemPool)
DONE Extend Device to hold index and change enum CUDA/CPU to be DeviceType and apply for it (compile)
DONE Update CUDAMemPool to apply for multiple devices (mallocs too)
DONE Rename the repo on github to gradcraft
DONE CUDA BRIDGE - ToNode and .to() without math on accumulate yet. Just call the dispatcher. CONTIGUITY FORCED. APPLY FOR OFFSET
DONE Write cuda kernels for filling with values (same or from a list)
DONE Integrate CUDAMemPool into Storage class for allocations (write first kernel for T(0) init value)
DONE Compile after first CUDA + get right results (fix the CUDAMemPool ptr** pass)
DONE DIM COALESCING ALGO HELPER
DONE INTEGRATE DIM FUSING post-broadcast
DONE Compile with dim fusing + get right results
DONE Write all functors in Functor::BIP::etc
DONE Write cuda kernels for UOOP, UIP, BIP, REDUCTIONS, BOOP
DONE Apply for scalars / fast paths on the GPU (scalars fall into the fast path) (all 4 main op types) (add swtiches to invoke with right functor the fast version)
DONE Write correct reduction kernels for CUDA (both fast and slow) + integrate them into cpu wrapper (with switch)
DONE Fast path for reduction with all axes on the CPU
DONE Template instantiate all
DONE Integrate CUDA into 
DONE Make CPU functors and GPU functors + integrate CPU ones
DONE Compile the code
DONE Write Max, Argmax kernels (CUDA) and apply (CPU)
DONE Integrate into dispatcher + frontend
DONE Printing for CUDA + item() for cuda
DONE Compile + test MaxNode + ArgMax
DONE Swap out current CPU testpark for CUDA testpark. Compile.
DONE Write alternative ctors for: ones, zeros, full, arange, randn, rand, weight inits
DONE Set up Async CUDA so GPUs dont lock each other
DONE .detach() method (breaks the link)
DONE Go through all current nodes. Check where during realize() exclusivity = mem saving (bin out of place, un out of place)
DONE SigmoidNode / TanhNode
DONE DivNode, SubNode, NegNode
DONE Optimize Div and Mul to use less memory + one grad equation in the other for Div
DONE GELUNode, SiLUNode
DONE Frontend for these 7 (Sub, Div, Neg, TanH, Sigmoid, SiLU, GeLU)
DONE Add InPlaceOp for Sub, Div
DONE ExpNode, LogNode + frontend
DONE Take care of types (requires) for stuff like Exp, Log, Tanh, Sigmoid, LUs
DONE Link up BLAS and cuBLAS + MatMulNode (if you do this its a giant W)
DONE Extend guard for matmul to only be instantiated for floating point ops
DONE Make infer_assert_device not force making a vector/copying
DONE Fix set_data to work on both CPU and CUDA 
DONE Compile + test out on both
DONE StackNode, SqueezeNode, UnsqueezeNode
DONE Frontend for these 3
DONE PowNode, SinNode, CosNode
DONE Frontend for these 3
DONE Write a functor mapper for cpu and cuda based on a scoped lambda (pass lambda that calls function)
DONE Fix the CUDA setDevice in apply_ functions to execute before the fast paths
DONE SoftmaxNode
DONE SqrtNode
DONE Make softmax free memory and without big branching falling back to default case
DONE LayerNormNode (remember about dependencies)
DONE free memory from cached results in all nodes during the backward pass when they are not used
DONE MinNode
DONE ArgMax, ArgMin Node (backward pass crash)
DONE retain_graph bool flag across Tensor backward, tensorstate, base Node class
DONE target_device in every node backward/forward
DONE Free copied mem for bwd if retain_graph = false during backward pass
DONE MSELossNode
DONE SoftmaxCrossEntropyNode restricted for floats
DONE SoftmaxCELN bwd for targets one-hot encoded
DONE DropoutNode
DONE BMMNode
DONE infer bmm meta func and BMMMeta 
DONE Parameter class
DONE no_decay in params
DONE named_parameters() / parameters() function
DONE load/get state dict from module
DONE make get state dict take in DEVICE
DONE make load state dict device-aware (calls .to into what its loading into)
DONE zero_grad() in module
DONE .to() in module
DONE base dir in NN/ folder for nn stuff
DONE Write initializers
DONE Restrict parameters to be dense
DONE Save checkpoint
DONE Load checkpoint
DONE Write linear layer
DONE Write ReLU layer
DONE Optimizer class to inherit from
DONE Write SGD with dispatch in place if grad available.
DONE Write SGDMomentum
DONE Write RMSProp
DONE Use .to() on scalars in optimizers
DONE Write AdamW (weight decay math after main param update)
DONE Func in optimizers to swap out params for params with new pointers (named)
DONE Sync function so that optim state updates to device params are on
DONE virtual functions inside base optimizer class that have default behavior (so SGD doenst crash)
DONE assert devices on parameters in optimizer (base class memeber func)
DONE Write state_dict() for optims
DONE Write load_state_dict(state_dict) for optims
DONE update lr function for optimizers (make lr be this->m_lr)
DONE write helper to infer shapes of embedding and embed vol
DONE write frontend for embeds
DONE write CPU apply embed and CUDA version + add into dispatcher
DONE write the embed node forward pass
DONE write the backprop for cpu and cuda + add into dispatcher
DONE write the backward for EmbedNode and wrap it up (integrate a .accum into tensor class)
DONE Write the Embedding module
DONE Compile
DONE Blast a giant test without __restrict__ on CUDA.
DONE add __restrict__ to cuda
DONE Blast again
DONE LayerNorm module
DONE Dropout module
DONE SPDA frontend free function
DONE Multi Head Attention!! 
DONE make_leaf() method
DONE Compile. Benchmark.
DONE AsyncToNode - uses unpageable cpu memory. Locks the memory on the CPU, queues up the transfer into CUDA, CPU moves to the next line
DONE .to_async frontend. only CPU to GPU
DONE num_params() - get param count
DONE Make tricks with all ops on matrices. Assert gradients are the same, assert math is the same as pytorch. Both devices
DONE Fix the BLAS illegal parameter 8 on tensors with strides [1, 1]
DONE overfit one batch of anything with every optimizer
DONE XOR test on SGD
DONE num_params method test
DONE save params, load and resume training. same with all optimizers
DONE RMSNorm naive frontend
DONE RMSNormNaiveNode
DONE RMSNorm kernel fast (contig strides / reduced dims)
DONE RMSNorm kernel strided
DONE Wrapper that fires forward kernels
DONE RMSNormFast backward fast kernel
DONE RMSNormFast backward strided version
DONE Force gamma dense in rmsnorm_fast()
DONE rmsnorm frontend dispatch (based on bool fast)
DONE RMSNormFastNode 
DONE Causal softmax dispatcher
DONE Causal softmax forward CUDA
DONE Causal softmax backward CUDA
DONE Causal softmax frontend
DONE Integrate causal softmax into SDPA fast frontend
DONE Fix LayerNorm module
DONE Write RMSNorm module
DONE PositionalEncoding module (slices indices)
DONE Check MHA if it fits sdpa func and the causal mask stuff not being done if cudafast=true
- dispatch scalar update + kernel
- Scheduler class
- CosineScheduler (with saving - scheduler class)
- clip_grad_norm, scales back, no explosions


- SwiGLU MLP module
- Transformer module
- GPT module
- ExtractPathesNode (backward pass crash) - purely util, like argmin/argmax - Im2Col or smth
- Compile
- add multithreading with OpenMP (no writing mutexes and shit)
- Create a diffusion transformer final proof project 
- Create a big GPT for coding himself (finetune maybe?)