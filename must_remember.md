1) You dont take by const reference& in operator+ since you would have to copy rvalues (say (a + b) * c), you cannot just take its data. You have to copy.
What i mean by stealing data is the std::move into AddNode. Its const, so it would be copied.
2) Tensors can share Storage, Storage AND m_op (alias) but never just m_op (makes zero sense)
3) m_op is a pointer, because otherwise (since its a childclass) would have its bits truncated. Memory would be only allocated for the base Node.
In addition to copy a Node you would have to copy a Tensor and to copy a Tensor you copy a node... (tho we dont copy it anywhere thats why its unique_ptr)
4) All nodes except CastNode return Tensor<T> and have parents of type Tensor<T>
5) All grads are contiguous because they are initialized contiguous. To ever assign anything in it, it must obey its contiguous strides. 
Everytime you accumulate a tensor, say, [5, 10, 32] of grad, it means "elem [0, 0, 0] here corresponds to grad of elem [0, 0, 0]"
It can have most crazy strides, but because of how accumulate_grad() works, it is added with strided indices, making the master buffer contiguous.
6) You cannot use lobotomize functions that allocate new storage during building a lazy graph (such as lobotomized_contiguous_alloc)
7) Reshaping is changing where contiguous tensor's rows break. If a tensor and its incoming grad are both contiguous after a view (reshape etc) operation, you can just take parents strides (1 to 1 mapping and strides would be the same)
If it is not (say, Transpose) and m_parent has weird strides, swapping contiguous tensor's strides to match is suicide. Grad strides transposed != parent strides (pre-transpose) cuz its not contiguous.
8) in-place optimization on AddNode if is_exclusive: If you got (a + b) + c, then (a + b) is an rvalue tensor. Nobody holds a map to it,
unlike if it was temp = a + b and res = temp + c. Since m_left is only ever used to produce end result during realize, you can safely edit it because
m_left inside node is the only place in universe where it exists. Nobody else will know, and addnode doesnt need it for anything else THEREFORE nobody needs it.
9) m_storage of TensorState must be initialized with empty shared_ptr and not nullptr. If c = a + b and c has a nullptr to storage, then d = c.reshape happens, 
and d keeps the nullptr. ReshapeNode has a copy of c, also with a nullptr. 
When realize happens, c storage is filled with data. ReshapeNode has none, d has none, but should. Everything blows up.
If there is a shared_ptr, its inside c, ReshapeNode, and d. Change immediately gets reflected.
10) Every apply function supposes memory is already allocated. It can be initialized for InPlace, or uninitialized for everytihng other (including Reduce)
11) Frontend asserts every single m_parent, m_left, m_right etc. is on the same device as tensor that will get result
12) Since frontend asserts that, accumulate_grad means result accumulates to m_parent, so grads are also on the right device. 
SHORTLY: Created lazily/view of Tensor<T> has the same device as member variables of its node.
13) If strides are [5, 10, 3, 2] and strides [60, 6, 1, 2] then you can fuse it into [50, 3, 2] with strides [6, 1, 2]. Because 60 = 6 * 10 (strides[i] = strides[i - 1] * shape[i - 1])
Why does it work? Because think about what dim 1 does. Goes 10 times, increments strided idx by 6. Total 60. When it finished its loop, dim 0 increments by 60. Then starts increasing by 6 again. 
If you have two tensors of the same shape, same strides but partially contiguous, you can instead of running strided loop just squash two dims into one, update strides.  
Less odometer strain.
14) The only reason fast paths dont blow up with __restrict is that we use apply_BIP smartly. If you BIP 2 contiguous tensors with the same storage (say slices that are contig)
they fall right into 
15) You could add further optimization in backward: because flow is Tensor.backward()->state.backward()->node.backward() and iterating over a list, then out_grad is destroyed
is destroyed as soon as node backward ends (every tensor state has a grad, and only one node). You COULD add checks like "Left & right both need out_grad to get calculated.
If left is already done and right needs grad, then ill reuse the out_grad buffer instead of allocating a new one". BUT: watch out if youre not modifying out_grad using out_grad.
This is where bugs slip (modifying storage with the same storage, __restrict and so on). Cannot modify storage using storage. 
But if its just out_grad * m_left? sure. Then unbroadcast and accumulate it after editing.
16) Another optimization: if BOOP both need grad, you allocate grad for one, then reuse it in the other as the buffer.
17) Look for places in BOOP where one grad relies on the other. You save calculations.
18) You pass orig_shape into accumulate_grad_matmul because MatMulNode holds fake flattened tensors. They are result of modified A or A.contiguous() with changed m_shape, m_strides. They do not have separate states. This means they share grad. So when you initialize grad, initialize it to A or A.contiguous() shape. Not the artificially flat one to satisfy BLAS
19) You can squeeze literally ANY tensor. Mathematically: nothing ever breaks. Thats because with dim_size = 1, the odometer loop for this
stride only ever shows 0. This means it literally does not ever matter.
20) During matmul, A is rendered flat. 
original -> optional: contiguous -> edit the shape/strides in place (flatten) -> pass into matmul 
MatmulNode holds the wrong shape logically. Therefore, before the reshaping happens, matmul func saves the real shape (batched, not flat) then passes it into MatMulNode.
Say you do C = A + B, matmul(C, D). C gets copied into matmul (alias) this alias now has m_op of AddNode(A, B). When this realizes, C realizes. The alias has a different shape than real C (edited in place). It may also have a contiguous node attached. It has the original C shape saved. It passes it into matmul.
21) There is no dispatch for ReduceOp::Mean because its handled in MeanNode (Sum then Div)
22) Clean only temporaries EXPLICITLY saved if retain_graph=false
23) You cannot wipe parents after backward pass because if parent is exclusive then you wipe its history
Say X depends on A and B. A is exclusive in some way. topo_order was built, its [X, A, B]. Bwd of X runs, this means bwd of AddNode. It wipes A that was exclusive. Then 
A->backward() runs but the tensor dereference crashes because it was wiped.