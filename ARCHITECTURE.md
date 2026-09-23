# Architecture of the `GradCraft` Autograd Engine

## Content
* [1. Devices](#1-devices)
* [2. Memory Pools](#2-memory-pools)
* [3. `Storage<T>` & RAII](#3-storaget--raii)
* [4. `TensorState<T>` & Computation Graph](#4-tensorstatet--computation-graph)
* [5. `Tensor<T>` Lightweight Wrapper](#5-tensort-lightweight-wrapper)
* [6. The Computational Graph & Frontend (Nodes & Callable Functions)](#6-the-computational-graph--frontend-nodes--callable-functions)
* [7. Autograd Engine & `.backward()` Flow](#7-autograd-engine--backward-flow)
* [8. Dispatchers](#8-dispatchers)
* [9. Fused Kernels](#9-fused-kernels)
* [10. DL Frontend (Module, Parameter, Optimizer)](#10-dl-frontend-module-parameter-optimizer)
* [Wrapping it All Up](#wrapping-it-all-up)

## 1. Devices
A `Device` struct holds enum `DeviceType` (which can be either `CPU` or `CUDA`), and its index (since you can have multiple GPUs accessible).  
There are a few helpers like the `==` or `is_cpu()` and `is_cuda()` to make writing easier  
This struct is all over the place. It guards trying to do math cross-device and tells the `dispatcher.cpp` whether to do math on the `CPU` or a specific `GPU`.

```cpp
enum class DeviceType {CPU, CUDA};

struct Device {
    DeviceType type;
    int32_t index;

    constexpr Device(DeviceType type = DeviceType::CPU, int32_t index = -1) : type(type), index(index) {}
}
```

## 2. Memory Pools
Since dynamically calling `malloc` and `cudaMalloc` inside a training loop is **_really, really expensive_**, `GradCraft` uses two singleton memory pools (`CPUMemPool` & `CUDAMemPool`).  
Both have an `allocate`, `free` and `clear` functions. `CUDAMemPool` uses `cudaMallocAsync` and `cudaFreeAsync` to avoid a CPU-GPU synchronization. 
Here is how the `Storage<T>` class requests memory from the pools:

```cpp
if (m_device.is_cpu()) {
    // rounds up to nearest 32-byte for SIMD alignment
    int64_t aligned_bytes = ((bytes + 31) / 32) * 32;
    m_data = static_cast<T*>(CPUMemPool::get().allocate(aligned_bytes));
}
else if (m_device.is_cuda()) {
    // gpu pool tracks and saves blocks by specific Device index, so gotta pass the device
    m_data = static_cast<T*>(CUDAMemPool::get().allocate(bytes, m_device));
}
```

### Key points
* **Memory aligned to 32 bytes:** CPU allocations are forced to be a multiple of 32 to enable SIMD (but who even trains on a CPU).
* **CUDA pool is device-aware:** `CUDAMemPool` groups blocks by both bytes and device, preventing, say, `CUDA:0` from getting a pointer meant for `CUDA:1`.

## 3. `Storage<T>` & RAII
To **NOT** deal with segfaults and deep copies, `GradCraft` wraps all pool allocations inside a RAII `Storage<T>` container. 

```cpp
template <typename T>
struct Storage {
    T* m_data = nullptr; 
    int64_t m_size = 0;
    Device m_device;

    // constructor controls memory allocation
    Storage(int64_t size, T init_val = T(0), Device device = Device(DeviceType::CPU), 
            bool allocate = true, bool fill = true);
            
    ~Storage(); // returns the m_data pointer to the correct memory pool slot

    // prevents accidental deep copies 
    Storage(const Storage&) = delete; 
    Storage& operator=(const Storage&) = delete; 
    
    Storage(Storage&& other); // can move Rvalues
};
```

### Key points
* **Strict RAII:** Copy constructors are deleted, so no `Storage<T>` can ever be duplicated by accident. Higher level objects (such as `TensorState` that will be mentioned soon) are forced to manage it via `std::shared_ptr`. When the ref count falls to 0, the destructor automatically returns the block back to memory pool.
* **Lazy evaluated engine (`allocate = false`):** `GradCraft` is lazily-evaluated. With this flag you can first build the DAG (directed acyclic graph) without requesting any RAM/VRAM until `.realize()` is called.
* **Saving memory bandwidth (`fill = false`):** Very often a tensor is created purely as a destination buffer for a math operation (addition of 2 tensors, `RMSNorm`, etc.). Forcing a `cudaMemset` to zero out memory that will be immediately overwritten is pointless and wastes PCIe/Memory bandwidth. Using `fill = false` hands you an instantly available uninitialized tensor you can write to.

## 4. `TensorState<T>` & Computation Graph (Read up to section 6. Don't try to understand raw `TensorState` and `Tensor` out of context without `Nodes` and frontend functions.)
In order to do backprop, the engine must track exactly how the loss was calculated. However, if we tied the computational history directly to the raw memory buffer (`Storage<T>`), creating a "view" (like transposing a matrix, reshaping, permuting...) would force a memory copy.

To solve this, we have a `TensorState`, which acts as a graph node, linking together memory and math history.

```cpp
template <typename T>
struct TensorState : public TensorStateBase {
    std::shared_ptr<Storage<T>> m_storage; 
    std::unique_ptr<Node<T>> m_creation_op; 
    
    bool m_is_realized;
    std::optional<Tensor<T>> m_grad = std::nullopt; 
    
    // ...
};
```
### Key points
* **History is ALWAYS exclusive (`std::unique_ptr<Node<T>>`):** A computational node (e.g. AddNode) uniquely belongs to the specific result it created. A `TensorState` does **NOT** share its history. If you have a tensor that was created via addition (TensorState has an AddNode) and then you transpose it, a **NEW** `TensorState` is created with a TransposeNode.
* **Storage buffers CAN be shared (`std::shared_ptr<Storage<T>>`):** While history is exclusive, we want to minimize memory usage. If you perform the transpose operation mentioned above, a new `TensorState` is created with a TransposeNode, but it points to the exact same `Storage<T>`. View operations READ but do not WRITE to memory, therefore they can just READ in whatever order it pleases from the same buffer.
* **ONE gradient:** The accumulated gradient (`m_grad`) lives inside the `TensorState`. This guarantees that if multiple tensors reference the exact same `TensorState`, their gradients safely accumulate into the same buffer during `.backward()`. Tensors only ever reference the same `TensorState` if they are aliases.

## 5. `Tensor<T>` Lightweight Wrapper

With memory isolated in Storage and history isolated in TensorState, the `Tensor<T>` class is just a wrapper around metadata of how we _look_ at the memory. 

```cpp
template <typename T>
class Tensor {
private:
    std::vector<int64_t> m_shape;
    std::vector<int64_t> m_strides;
    int64_t m_offset;
    
    std::shared_ptr<TensorState<T>> m_state;
    bool m_requires_grad;
    // ...
};
```

### Key points
* **Tensor aliasing `shared_ptr<TensorState<T>>`:** When you copy a tensor (`Tensor B = A` or copy assignment), no memory is copied and no new nodes are created. `B` simply copies the shared_ptr to `A`'s `TensorState`. They are ideal aliases - they share the same memory, the same mathematical history and the same gradient. When one is realized, the other is also realized. If both are used in math and `.backward()` is called, they both accumulate to the same gradient buffer.
* **Zero-copy shape/stride/offset manipulation:** The `m_shape`, `m_strides` and `m_offset` vectors define the "view". Operations like `.slice()`, `.unsqueeze()` or `.permute()` do not touch the GPU memory. They simply create a new `Tensor`, calculate the new mathematical shape/strides/offset, assign it a new `TensorState` (to track the view operation for the autograd) and point it at the exact same `Storage`.
* **Need For Gradients (`m_requires_grad`):** The engine tracks which tensors have to be included during calculation of gradients at the graph-building stage (eagerly).

## 6. The Computaional Graph & Frontend (Nodes & Callable Functions)

To make the terminology clear: 
* A frontend function (what I call it) is a function called on one or more `Tensor` objects. It is what the user actually calls on his tensors such as `A.reshape()` or `matmul(A, B)`. They always return a `Tensor<T>`.
* A lobotomized `Tensor` is a `Tensor` with a `TensorState` that does not have an `m_creation_op` (history) and `m_requires_grad = false`. It is a `Tensor` that is unaware that any graph exists. It has `m_strides`, `m_shape`, `m_offset` and may or may not have a full storage. `Lobotomized = has no idea about graph`. (note: I randomly used this name early in the project and it stuck to me)  

Let's look at one frontend function that creates a lazy `Tensor<T>`

```cpp
// lets assume we ran B = A.tanh()
template <typename T>
Tensor<T> Tensor<T>::tanh() const { 
    Tensor<T> result = Tensor<T>(m_shape, m_requires_grad, lazy, this->device()); // create a lobotomized lazy tensor WITHOUT any Node<T> inside of its TensorState<T> and without any memory allocated in its Storage<T>
    result.m_state->m_creation_op = std::make_unique<TanHNode<T>>(*this); // overwrite the m_creation_op that is nullptr in the TensorState with a pointer to a TanHNode that holds A. The Tensor is not lobotomized anymore.
    return result; // return the lazy tensor with graph history inside of it. 
}
```

Now what does `TanHNode` look like? (I stripped the class to a minimal .realize() function. About `.backward()` later.)

```cpp
template <typename T>
class TanHNode : public Node<T> {
    private:
        Tensor<T> m_parent; // the *this (that is Tensor A) is now held in m_parent. It is an alias with the same TensorState pointer as A. (see the first comment below)
    public:
        TanHNode(Tensor<T> parent) : m_parent(std::move(parent)) {} // the constructor takes in parent BY VALUE. This means it has the same shape vector, strides vector and the TensorState pointer.

        Tensor<T> realize() override { // Every single nodes' .realize() function returns a Tensor<T>
            m_parent.realize(); // first realizes its parent. You cannot calculate your output if you don't know your input. I will explain Tensor.realize() in a moment.
            Device target_device = m_parent.device();

            Tensor<T> result = Tensor<T>(m_parent.shape(), target_device, uninitialized); // Create a Tensor with uninitialized memory that we will write to. Absolutely lobotomized.
            dispatch(target_device, UnaryOp::TanH, result, m_parent); // run the TanH calculations on the correct device accumulating into the correct Tensor (result)

            return result; // returns a Tensor<T> with the result calculated and sitting inside its Storage<T> buffer
        }
};
```
### Key points
* Generally nodes are what bridges the `TensorState` of the inputs, to the `TensorState` of the result. That is because `Tensor` holds a `TensorState` which holds a `Node` which holds a `Tensor` which holds a `TensorState` which holds a `Node`... down to a `Tensor` _leaf tensor_ - a `Tensor` which has no `Node` but has numbers inside of it already, what means it does not have to know how to calculate itself.
* Whenever a frontend function is used on a tensor, the engine does not compute the result immediately. Instead, it creates a lazy result (no allocation) and attaches a `Node` to its `TensorState` with a parent passed (or multiple parents).  
* Whenever `Node.realize()` is called and `Node` is **NOT** a view node, a fresh **NON-lazy, lobotomized** `Tensor` is created, and an operation is dispatched to fill its `Storage<T>`. Then its returned.

After the whole graph has been built, then `.realize()` can be called on a `Tensor` which will automatically resolve what it has to calculate and in what order. How?

```cpp
template <typename T>
void Tensor<T>::realize() {
    if (m_state->m_creation_op != nullptr && m_state->m_is_realized != true) { // is not leaf and wasnt realized yet
        Tensor computed_result = m_state->m_creation_op->realize(); // reach into my state and grab my Node. Then get a lobotomized result of the node with freshly filled Storage<T>
        if (m_state->m_storage != computed_result.m_state->m_storage) { // if the shared_ptr to Storage that the Node returned is not my current shared_ptr to Storage (view nodes do that!)
            std::swap(m_state->m_storage->m_data, computed_result.m_state->m_storage->m_data); // swap the pointer IN the Storage TO data, not TO the storage.
        }
    }

    m_state->m_is_realized = true; // so we dont realize() twice (reflected across multiple aliases)
}
```

### About the `std::swap(m_state->m_storage->m_data, computed_result.m_state->m_storage->m_data)` line.  

Multiple `TensorStates` can look at the same `Storage` via `shared_ptr`. They all expect to have the same data and the same `shared_ptr`. We have to accept this assumption for now, because view nodes exploit that heavy. More on that shortly.

### Key points
* Leaf tensors are NOT realized since they are filled with numbers at the start.
* We steal the `Storage` pointer from the `Tensor` `computed_result` of `Node.realize()`
* View nodes `.realize()` result Tensors have the same shared_ptr pointer to `Storage` as the Tensor we are realizing. More on that in a second.
* Since Tensors share `TensorState` realization of one realizes all.
* Every single `TensorState` has `bool m_is_realized` so that aliases dont realize many times wasting memory

Here is an `AddNode` and `operator+` for reference for the stuff below (also stripped to bare minimum)

```cpp

template <typename T>
auto operator+(Tensor<T> left, Tensor<T> right) { // frontend function
    bool requires_grad = p_left.m_requires_grad || p_right.m_requires_grad; // do the parents require grad dL/dleft or dL/dright? if so, result also requires. Thats calculus tho, not CS.
    Tensor<T> new_tensor = Tensor<T>(target_shape, requires_grad, lazy, target_device); // create a lobotomized lazy tensor with no memory allocated
    new_tensor.m_state->m_creation_op = std::make_unique<AddNode<T>>(std::move(p_left), std::move(p_right), std::move(target_shape)); // un-lobootmize it by attaching graph to its TensorState
    return new_tensor;
}


template <typename T>
class AddNode : public Node<T> {
    private:
        Tensor<T> m_left;
        Tensor<T> m_right;
        std::vector<int64_t> m_target_shape;
    public:
        AddNode<T>(Tensor<T> left, Tensor<T> right, std::vector<int64_t> target_shape) : m_left(std::move(left)), m_right(std::move(right)), m_target_shape(std::move(target_shape)) {}
        
        Tensor<T> realize() override {
            m_left.realize(); // make sure parents are realized so that math can be done
            m_right.realize();

            Device target_device = m_left.device();

            Tensor<T> result = Tensor<T>(m_target_shape, target_device, uninitialized); // lobotomized, allocated tensor
            dispatch(target_device, BinaryOp::Add, result, m_left, m_right); // fill the tensor

            return result;
        }
};
```
Lets track the flow of this code:

```cpp
Tensor<T> A = Tensor<T>(... initialize it to some numbers);
Tensor<T> B = Tensor<T>(... also initialize to some numbers);

Tensor<T> C = (A + B).tanh();
C.realize();
```

1. `A + B` creates an Rvalue `Tensor` with an `AddNode` in its' `TensorState` which holds `A` in `m_left` and `B` in `m_right` BY VALUE. We name this Tensor `temp_sum`.
2. `temp_sum.tanh()` runs. It creates a `Tensor` with a `TensorState` with a `TanHNode` which holds `temp_sum` inside `m_parent`. This `Tensor` is our `C`.
3. `C.realize()` runs and triggers `TanHNode.realize()`.
4. `TanHNode.realize()`triggers `m_parent.realize()` which means `temp_sum.realize()`
5. `temp_sum.realize()` triggers `AddNode.realize()` which triggers `m_left.realize()` and `m_right.realize()` which means `A.realize()` and `B.realize()` but they are leaf tensors, so they return early.
6. `AddNode.realize()` creates a `Tensor result` and fills it with data via `dispatch`
7. `AddNode.realize()`returns the brand new `Tensor` to `temp_sum.realize()` as `computed_result`
8. The `shared_ptr` is different and `temp_sum` inner `Storage` pointer is swapped with `computed_result` one.
9. `TanHNode.realize()` continues executing. It creates a `Tensor result` and fills it with numbers via `dispatch`.
10. `TanHNode` returns the brand new `Tensor` to `C.realize()` as `computed_result`.
11. The `shared_ptr` is different and `C` inner `Storage` pointer is swapped with `computede_result` one.

...and `C` is now filled with correct values.

### Key points:
* Every single `Node` must first realize its parents.
* The math order figures itself out. If `AddNode` runs, it makes sure `m_left` and `m_right` are realized, so that it has something to do math on in the `dispatch`.

Now lets look closer at the view nodes, why we have the check in `Tensor.realize()` and why we swap inner `Storage` pointers and not `shared_ptr<Storage>`. 

Here is reference code for transposing.

```cpp

template <typename T>
Tensor<T> lobotomized_transpose_view(const Tensor<T>& source, int64_t dim0, int64_t dim1) {

    std::vector<int64_t> new_shape = source.m_shape;
    std::vector<int64_t> new_strides = source.m_strides;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    // create a Tensor that is lobotomized (m_creation_op = nullptr, m_requires_grad = false) BUT holds the same Storage as Tensor<T>& source.
    // It also has NEW shape and NEW strides.
    Tensor<T> result = Tensor<T>(std::move(new_shape), std::move(new_strides), source.m_offset, source.m_state->m_storage, false);
    
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::transpose(int64_t dim0, int64_t dim1) const { // frontend function
    Tensor<T> result = lobotomized_transpose_view(*this, dim0, dim1); // the result tensor holds the exact same Storage shared_ptr as *this
    result.m_state->m_creation_op = std::make_unique<TransposeNode<T>>(*this);  // create a TransposeNode with *this and some data for backward pass
    result.m_requires_grad = m_requires_grad;
    return result; // core: if B = A.reshape() then even before B.realize(), B already holds the same shared_ptr<Storage<T>> as A.
}

 template <typename T>
class TransposeNode: public Node<T> {
    private: 
        Tensor<T> m_parent;
        int64_t m_dim0;
        int64_t m_dim1;

    public:
        TransposeNode(Tensor<T> parent, int64_t dim0, int64_t dim1) : m_parent(std::move(parent)), m_dim0(dim0), m_dim1(dim1) {}

        Tensor<T> realize() override {
            m_parent.realize(); // first realize parent
            return m_parent; // Since this is called when doing B.realize() and B shared_ptr is same as A (and A = m_parent) then just return the m_parent.
        }
};
```
We will track these lines:

```cpp
Tensor<T> A = Tensor<T>(... initialize it to some numbers);
Tensor<T> B = Tensor<T>(... also initialize to some numbers);

Tensor<T> C = (A + B).transpose();
C.realize();
```

1. `A + B` creates an Rvalue `Tensor` with an `AddNode` in its `TensorState` which holds `A` in `m_left` and `B` in `m_right` BY VALUE. We name this `Tensor` `temp_sum.`
2. `temp_sum.transpose()` runs. It calls `lobotomized_transpose_view`. This creates a new `Tensor` with swapped shapes and strides, but hands it the exact same `shared_ptr<Storage>` as `temp_sum`.  
3. Back in the frontend `.transpose()`, we attach a `TransposeNode` to this new tensor's state, holding `temp_sum` inside `m_parent`. This new `Tensor` is our `C`.  
4. `C.realize()` runs and triggers `TransposeNode.realize()`.  
5. `.realize()` immediately triggers `m_parent.realize()`, which means `temp_sum.realize()`.  
6. `temp_sum.realize()` triggers `AddNode.realize()`, which triggers `A.realize()` and `B.realize()` (they are leaves, so they return early).  
7. `AddNode.realize()` creates a `Tensor result`, allocates physical memory, fills it via `dispatch`, and returns it as `computed_result`.  
8. `temp_sum.realize()` receives `computed_result`. Because `temp_sum` is lazy, its original raw memory pointer was empty. It executes `std::swap(m_state->m_storage->m_data, computed_result.m_state->m_storage->m_data)`.  
9. Core: Because `C` was constructed to hold the exact same `shared_ptr<Storage>` as `temp_sum`, `C` instantly "sees" this new computed memory.
10. `TransposeNode.realize()` resumes. It dispatches zero math and allocates zero memory. It simply returns `m_parent` (`temp_sum`) directly.  
11. `C.realize()` receives `temp_sum` as its `computed_result`. It checks if `m_storage != computed_result.m_storage`. Since they literally share the same storage, this evaluates to false. No pointers are swapped, and `C` is marked as realized.  

### Key points
* If `temp_sum` simply overwrote its `shared_ptr<Storage>` with the new one from `computed_result`, `C`'s `Storage` would be left pointing to the old uninitialized dummy storage. By swapping the inner raw `m_data` pointer inside the shared `Storage` object, the memory update instantly propagates to every view `Tensor` in the graph that uses the same `Storage`. This is how we get allocation-free views.

Knowing how nodes and frontend interact, here is the general three-function contract that every `Node` has to satisfy.

```cpp
template <typename T>
class Node {
    public:
        virtual Tensor<T> realize() = 0;

        virtual void backward(const Tensor<T>& out_grad, bool retain_graph) = 0; // accumulates grad to parents

        virtual std::vector<TensorStateBase*> get_input_states() = 0; // enables establishing the order of computing gradients (what Node when)
};
```

## 7. Autograd Engine & `.backward()` Flow.

It is important to get one param out of our way - `bool retain_graph`. It is passed to `Node.backward()` and prevents wiping of intermediate result that the specific `Node` saved during `.realize()`. That is because the user may not want to clear the graph after the first `.backward()` and will want to run `.backward()` again. Without intermediate results it is impossible. Simple as that, I just dont wanna mention it later.

I think it will be better to show the code and **then** talk talk about how backward pass works out fully. 

First - you have already seen `TensorStateBase` class. That is just the base **UNTEMPLATED** class around `TensorState`. It declares three virtual methods, and since `TensorState` inherits publicly, a `TensorState` **is** a `TensorStateBase`. I needed this wrapper, because we use `std::vector` to hold the order in which to evaluate `.backward()` with mathematical correctness. The issue is that `std::vector` can hold only one type of variables and in the graph we can have a `TensorState<int64_t>` (say, indices) and a `TensorState<float>` (real fp32 calculations). Those are 2 different types.

Simple code:
```cpp
struct TensorStateBase {
    public:
        virtual std::vector<TensorStateBase*> get_dependencies() const = 0;

        virtual void backward(bool retain_graph) const = 0;

        virtual void clear_grad_if_non_leaf() = 0;
};
```

Here is how `TensorState` overrides these 3 virtual functions:

```cpp
template <typename T>
    struct TensorState : public TensorStateBase {
        std::vector<TensorStateBase*> get_dependencies() const override {
            if (m_creation_op == nullptr) {
                return std::vector<TensorStateBase*>();
            }
            return m_creation_op->get_input_states(); // Node returns its parents' TensorStates in an std::vector
        }

        void backward(bool retain_graph) const override {
            if (m_creation_op != nullptr && m_grad.has_value()) { // if I have dL/dmyself and I was created via a Node
                m_creation_op->backward(m_grad.value(), retain_graph); // I force the Node to evaluate its .backward() and accumulate grad to its parents TensorStates.
            }
        }

        void clear_grad_if_non_leaf() override {
            if (m_creation_op != nullptr) { // if I am NOT a leaf (something created me)
                m_grad = std::nullopt; // wipe my gradient
            }
        }
    };
```

### Key points:
* A `TensorState` can get its dependencies (which `TensorStates` were used in creating it) via `get_dependencies()`.
* `TensorState.backward()` passes its fully accumulated gradient to a `Node` that created it. The `Node` calculates correct gradients for both parents and calls `Tensor.accumulate()`.
* A `TensorState` can have its `m_grad` set to `std::nullopt`. That's because the `Tensor` holding it may not have required gradients (like `indices` or `targets`).

Moving on, lets look at the `AutogradEngine::build_topo(TensorStateBase* root)`

```cpp
class AutogradEngine {
    public:
        static void visit(TensorStateBase* current, std::unordered_set<TensorStateBase*>& visited, std::vector<TensorStateBase*>& topo_order) {
            if (visited.contains(current)) {
                return;
            }

            visited.insert(current);

            for (TensorStateBase* new_root : current->get_dependencies()) {
                visit(new_root, visited, topo_order);
            }

            topo_order.push_back(current);
        } 

        static std::vector<TensorStateBase*> build_topo(TensorStateBase* root) {
            std::unordered_set<TensorStateBase*> visited;
            std::vector<TensorStateBase*> topo_order;
            
            visit(root, visited, topo_order);

            std::reverse(topo_order.begin(), topo_order.end());
            return topo_order;
        }
};
```

### Key points
* You pass in a `TensorStateBase* root` and it recursively visits its dependencies, tracking their order.
* Dependencies are added to the list first, results last. If` Y = A + B` then the order (NOT reversed) is: `[A, B, Y]` or `[B, A, Y]`. What was created is LAST.
* If `Z` depends on any number of variables, which then may depend on any number of variables, it will always be AFTER its dependencies. This holds because `build_topo()` adds the result to the list AFTER a recursive call to `visit()`.  

One line in particular is important: `std::reverse(topo_order.begin(), topo_order.end());`.  

If `Y = A + B`, the reversed order is: `[Y, A, B]` or `[Y, B, A]`. Dependencies are evaluated last, results are evaluated first.  
Why must we strictly reverse the list? It comes down to the multivariable chain rule. Let's look at an example where a variable branches out:  

`A, B, C` are leaf tensors.  
`X = A + B`  
`Y = X + C`  
`L = X * Y`  

`X` is used twice - once to compute `Y`, and once to compute the final loss `L`.

To correctly backpropagate gradients down to `A` and `B`, the engine relies on `X.backward()`. But before `X` can calculate $\frac{\partial L}{\partial X} \times \frac{\partial X}{\partial A}$, the gradient buffer inside `X`'s `TensorState` MUST contain the fully accumulated value of $\frac{\partial L}{\partial X}$. Because `X` branched into both `L` and `Y`, its total gradient is the sum of the gradients flowing backward from both of those paths.

If the order wasn't results first, dependencies last, `X.backward()` might execute after receiving the gradient from `L`, but before receiving the gradient from `Y`. If that happened, `X` would pass an incomplete, wrong gradient down to `A` and `B`. By running `std::reverse()`, we guarantee that a node (like `X`) will only execute its `.backward()` step strictly after every single node that depends on it (`L` and `Y`) has completely finished executing.  

1. `L` runs, pushing gradients into `X` and `Y`.
2. `Y` runs, pushing gradients into `X` and `C`.
3. `X` runs. Its `m_grad` is now fully populated. It calculates the correct chain rule and pushes the final gradients into `A` and `B`.

When correct order is guaranteed, calculating the entire backward pass is just a flat loop:

```cpp
template <typename T>
void Tensor<T>::backward(bool retain_graph) {
    // build the mathematically correct execution order
    std::vector<TensorStateBase*> topo_order = AutogradEngine::build_topo(m_state.get());
    
    // set dL/dL to be 1.0
    if (!m_state->m_grad.has_value()) {
        m_state->m_grad = Tensor<T>::ones(m_shape, device());
    }
    
    // loop and push gradients down
    for (TensorStateBase* current : topo_order) {
        current->backward(retain_graph);
        current->clear_grad_if_non_leaf();
    }
}
```

When `current->backward()` is called, it (`TensorState`) launches `.backward()` of the `Node` that created it. For example, look at how an `AddNode` accumulates gradients to its parents:

```cpp
void backward(const Tensor<T>& out_grad, [[maybe_unused]] bool retain_graph) override {
    if (m_left.requires_grad()) {
        m_left.accumulate_grad(unbroadcast_grad(out_grad, m_left.shape()));
    }
    if (m_right.requires_grad()) {
        m_right.accumulate_grad(unbroadcast_grad(out_grad, m_right.shape()));
    }
}
```

Notice it does not replace the parents' gradients, but accumulates. This is what fixes the issue shown above with variables branching out.  

What does `Tensor.accumulate_grad()` look like?  

```cpp
template <typename T>
void Tensor<T>::accumulate_grad(const Tensor<T>& incoming_grad, bool is_sub) {
    if (!m_requires_grad) {return;}
    
    if (!m_state->m_grad.has_value()) {
        // first time seeing gradient (from L) - create gradient buffer and copy over
        Tensor<T> local_grad = Tensor<T>(m_shape, target_device, uninitialized);
        dispatch(target_device, UnaryOp::Identity, local_grad, incoming_grad);
        m_state->m_grad = std::move(local_grad);
    }
    else {
        // next gradients (Y) - dispatch an addition operation to m_grad
        dispatch(target_device, BinaryOpInPlace::Add, m_state->m_grad.value(), incoming_grad);
    }
}
```

A word about `clean_grad_if_non_leaf()`:  

During the loop we run `current->clean_grad_if_non_leaf()`. Gradients of intermediate results can take gigabytes of VRAM, so they are wiped once their duty is done (their parent accumulated correct grad using it). They simply go back to memory pool to be used again.

Also a word about `unbroadcast_grad()`:  

Operations often broadcast smaller tensor to match larger ones (say, `gamma` in `RMSNorm`). To preserve the chain rule, output gradients are summed along all axes where broadcasting took place. Its used all over the place, `AddNode`, `MulNode`, essentially everywhere where broadcasting can take place.

## 8. Dispatchers.

The `Node` classes know calculus, but they are blind to hardware / loops / how to execute the math. To actually execute math, they call `dispatch()`. This abstraction allows switching model devices via `.to()` without having to change any logic - the dispatcher will take care of what to run.

The dispatchers are essentially just a router for math. They take a few tensors and an operation to run. For instance, `Binary-Out-Of-Place` ops (BOOP) require three tensors - `left`, `right` and `result`. `Binary-In-Place` ops require just `left` and `right`.

Importantly, they check whether to execute math on the `CPU` or `CUDA`. 

Here is an example of a dispatcher, that executes op on `left` and `right` and writes to `out`. 

```cpp
template <typename T>
inline void dispatch(Device device, BinaryOp op, Tensor<T>& out, const Tensor<T>& left, const Tensor<T>& right) {
    if (device.is_cpu()) {
        cpu_mapper::map_boop<T>(op, [&](auto functor) {CPUBackend::apply_binary_out_of_place(out, left, right, functor);});
    }
    else if (device.is_cuda()) {
        CUDAMath::apply_binary_out_of_place(out, left, right, op);
    }
}
```

You can plug in any `op` (like `BinaryOp::Add`) and the dispatcher will find a lambda that takes two numbers and returns their sum. 

The math backend is highly optimized for common operations - such as when adding two contiguous tensors of the same shape - both `CPU` and `CUDA` just blast a quick 1D loop, instead of a slow odometer loop.  

Not all dispatchers are CPU-available though. I wrote a few fast kernels for `CUDA` to fuse operations, instead of dispatching multiple primitives. (more in the next section)

```cpp
template <typename T>
inline void dispatch_rmsnorm_forward(/* a lot of params */) {
    if (device.is_cpu()) {
        throw std::runtime_error("Tried running RMSNormFast forward on the CPU.");
    }
    else if (device.is_cuda()) {
        CUDAMath::apply_rmsnorm_forward(out, inv_rms, parent, gamma, red_meta, normalized_shape, eps);
    }
}
```

To not clutter `Nodes` with `is_cuda()` and `is_cpu()` checks, I simply added few `Nodes` like `SoftmaxCrossEntropyFastNode` or `RMSNormFastNode` that are only attached if we are using `CUDA`. These call the CUDA-unique dispatchers.

## 9. Fused Kernels.

Some deep learning layers / operations are just SO expensive to do primitively. Memory bandwidth is the usual bottleneck in DL, so I wrote dedicated `CUDA` kernels.  

Naive nodes just blast `dispatch()` X times. ``RMSNormNaiveNode` calls them 9 _**(NINE!!!)**_ times just in the forward pass. Its **beyond saving** bro.

```cpp
// piece of the RMSNormNaiveNode.realize() code
dispatch(target_device, UnaryOp::Square, scratchpad, m_parent);
Tensor<T> inv_rms = Tensor<T>(m_red_meta.temp_shape, target_device, uninitialized); // (B, T, 1)
dispatch(target_device, ReduceOp::Sum, m_red_meta, inv_rms, scratchpad);
dispatch(target_device, BinaryOpInPlace::Div, inv_rms, Tensor<T>(m_red_meta.reduced_vol, target_device));
dispatch(target_device, BinaryOpInPlace::Add, inv_rms, Tensor<T>(m_eps, target_device));
dispatch(target_device, UnaryOpInPlace::Sqrt, inv_rms);
dispatch(target_device, BinaryOpInPlace::IDiv, inv_rms, Tensor<T>(static_cast<T>(1.0), target_device));
```

Fast nodes call dispatch once, and run a single `CUDA` kernel.

```cpp
Tensor<T> result = Tensor<T>(m_parent.shape(), target_device, uninitialized);
Tensor<T> reshaped_gamma = lobotomized_reshape_view(m_gamma, m_normalized_shape);

dispatch_rmsnorm_forward(/* lots of params */);
```

The `RMSNorm` `CUDA` kernel supports both contiguous and not contiguous inputs entering it. `RMSNorm` code for forward, backward for both variants is a total of over 400 lines, so I'm not going to paste it here. It's all in `gradc/backend/cuda/kernels_math.cu`. Here is a small piece tho that calculates squared sum.

```cpp
for (int64_t i = tid; i < reduced_vol; i += blockDim.x) {
    T val = parent_row[i];
    thread_sq_sum += val * val;
}

__shared__ T s_sum[256]; // hardcoded 256 threads

s_sum[tid] = thread_sq_sum;
__syncthreads();

for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        s_sum[tid] += s_sum[tid + s];
    }
    __syncthreads();
}
```

Benchmarks show that the fused version of `RMSNorm` is **95.7x!!!** _**(NINETY-FIVE POINT SEVEN bro)**_ times faster. Its all in `BENCHMARKS.md`

## 10. DL Frontend (Module, Parameter, Optimizer).

Last section (I think?). All the math and states are now being wrapped in a clean PyTorch-like API so its easy to just put LEGO blocks together.

* `Parameter<T>`: This is just a `Tensor` wrapper that enforces strict rules. It sets `requires_grad = True` and it must be dense (contiguous, no offset). It holds a `no_decay` flag. If you set it to `true`, `Optimizer` skips applying `decay` to the param.
* `Module<T>`: It holds `std::vector` of submodules and parameters. `named_parameters()` recursively walks the tree, picking up all `Parameter<T>*` and putting them in a hash map. It allows checkpointing via `state_dict` and `load_state_dict`, zeroing gradient via `zero_grad()` and moving giant amount of weights via `.to(Device)`.
* `Optimizer<T>` (There are 4 of them, but `AdamW` is the most interesting): It takes a map of parameters and updates them. Actually `AdamW` has its own `dispatch_adamw_step()` and a `CUDA` kernel. Naive implementation is just too slow. (about 80 times slower)

```cpp
if (m_cuda_fast && target_device.is_cuda()) {
    for (auto& [name, p_ptr] : this->m_named_params) {
        if (!p_ptr->grad().has_value()) { continue; }
        dispatch_adamw_step(target_device, p_ptr->tensor(), m_first_moment[name], m_second_moment[name], p_ptr->grad().value(), this->m_lr, m_beta1, m_beta2, m_beta1_exp, m_beta2_exp, m_weight_decay, m_eps, p_ptr->no_decay());
    }
    return;
}
```

## Wrapping it All Up.

As I am writing this there is a 90 000 000 param LLM called **MALLMOC** (LLM + MALLOC = MALLMOC) training on my RTX 3090 GPU.

Building `GradCraft` was a brutal, incredibly rewarding and teaching experience. It took over 10,000 lines of C++, hundreds of lines of CUDA kernels and hours of debugging segfaults.

To prove that the architecture actually works and it trained *something* here is MALLMOC-90 (once it finished training). It speaks Python (not C++ because it's way more complex).

`PROMPT:`
```py
def reverse_string(
```
`PROMPT + ANSWER:`
```py
def reverse_string(string: str):
    return ''.join(reversed(string))
```

Summing up: you could just do `[::-1]` or you could write a 12,000 line deep learning framework with your own kernels, and train a 90M clanker to do the job for you. For me the choice was obvious.

Thanks for reading. All the code is in the repo.
