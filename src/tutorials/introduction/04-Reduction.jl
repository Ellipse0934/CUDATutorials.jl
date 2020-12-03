#=
# Reduction

The `reduce` function takes in a binary operator `⊕` and a ordered collection, applying the operator to that collection effectively reducing it to one final value.

For example, the operator ⊕ can be `minimum` and the collection can be an array of Integers.
⊕ can also be `addition` or `xor` and the collection may represent any type of object as long as the operator makes sense in the context.
=#

using CUDA

a = rand(10)

#-

reduce(min, a)

#-

reduce(*, a)

#-

@doc reduce(*, a)

#=
Writing `reduce` for a CPU is quite straightforward with a single `for-loop`. We will focus on writing a reduction for a linear array in this tutorial. We will iteratively develop a performant version using everything we have learnt in the previous tutorials.

---

# Reduction 1 : Divide and Conquer

The first step is to write something that works on the GPU. If we were given a dual-core machine and expected to parallelize this we would split the input array into two halves and feed each half into a different CPU. Similarly the best hint is to use the [divide-and-conquer](https://en.wikipedia.org/wiki/Divide-and-conquer_algorithm) approach.
By envisioning the reduction process as a binary tree we get: 

![reduction-1](../assets/reduction1.png)

It would be a good exercise to try to write model the above process in pseudocode.

One such approach is:


The only issue with this approach on the GPU is that after each step we need to synchronize which won't be possible with arrays which span over a single thread block (1024 threads is the maximum threads in a block). Hence, we will have to use a recursive approach.

Assume for now we have $1024$ threads per block and process one element per thread. If we have $2048$ threads then we will run our algorithm with two thread blocks, storing the results in an intermediate array. After our first kernel is done we will perform a reduction on the intermediate array. And if we have an array whose length is greater than $1024^2$ we will have another level of recursion. If there are $1024*1024 + 1$ elements then the $1^{st}$ level of reduction will return an intermediate array of size $1025$ which will take another level of recursion to process.
=#

function reduction1(op, a::CuArray)
    threadsPerBlock = 1024
    len = length(a)
    
    sums = similar(a, cld(len, threadsPerBlock))
    
    blocks = cld(len, threadsPerBlock)
    shmem = sizeof(eltype(a))*threadsPerBlock
    @cuda shmem=shmem threads=threadsPerBlock blocks=blocks reduction1_kernel(op, a, sums)
    
    ## Recursively call reduction for larger arrays
    if length(sums) > 1
        return reduction1(op, sums)[1]
    end
    
    CUDA.@allowscalar return sums[1]
end

#-

function reduction1_kernel(op, a, sums)
    shmem = @cuDynamicSharedMem(eltype(a), (blockDim().x, ))
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index = threadIdx().x
    len = blockDim().x
    
    ## Adjust length for the last block
    if blockIdx().x == gridDim().x
        len = mod1(length(a), blockDim().x)
    end
    
    if tid <= length(a)
        shmem[index] = a[tid]
    else
        return
    end
    sync_threads()
    
    steps = floor(Int, CUDA.log2(convert(Float32, len)))
    for i = 0:steps
        if mod(index - 1, 2^(i + 1)) == 0 && (index + 2^i) <= len
            shmem[index] = op(shmem[index], shmem[index + 2^i])
        end
        sync_threads()
    end
    
    if index == 1
        sums[blockIdx().x] = shmem[1]
    end
    return
end

#-

a = CUDA.ones(1025);
reduction1(+, a)

#-

a = CUDA.rand(100_000);
println(reduction1(+, a),"  ",reduce(+, a))

#=
The two results above not being exactly equal is expected since IEEE floats are neither associative nor commutative. Since associativity is tough to achieve on a parallel algorithm we can expect some deviation.

---

# Reduction 2 : Strided Index

One problem with the last reduction was divergent branching, for example thread three is active for exactly one computation (`a[3] op a[4]`) and is never used again. Because GPU's operate warpwise we want to use all the resources of a warp instead of a small subset. When threads in a warp do different things it has diverged and it's efficiency drops. In this case with each iteration half the number of threads go inactive.

A simple fix is to change the way threads map to the elements by using a strided index

![reduction-2](../assets/reduction2.png)
=#

function reduction2_kernel(op, a, sums)
    shmem = @cuDynamicSharedMem(eltype(a), (blockDim().x, ))
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index = threadIdx().x
    len = blockDim().x
    
    ## Adjust length for the last block
    if blockIdx().x == gridDim().x
        len = mod1(length(a), blockDim().x)
    end
    
    if tid <= length(a)
        shmem[index] = a[tid]
    else
        return
    end
    sync_threads()
    
    steps = floor(Int, CUDA.log2(convert(Float32, len)))
    for i = 0:steps
        stride = 2^i
        sindex = 2*stride*(index - 1) + 1
        if sindex + stride <= len
            shmem[sindex] = op(shmem[sindex], shmem[sindex + stride])
        end
        sync_threads()
    end
    
    if index == 1
        sums[blockIdx().x] = shmem[1]
    end
    return
end

#-

function reduction2(op, a::CuArray)
    threadsPerBlock = 1024
    len = length(a)
    
    sums = similar(a, cld(len, threadsPerBlock))
    
    blocks = cld(len, threadsPerBlock)
    shmem = sizeof(eltype(a))*threadsPerBlock
    @cuda shmem=shmem threads=threadsPerBlock blocks=blocks reduction2_kernel(op, a, sums)
    
    ## Recursively call reduction for larger arrays
    if length(sums) > 1
        return reduction2(op, sums)
    end
    
    CUDA.@allowscalar return sums[1]
end

#-

a = CUDA.ones(100_000);
reduction2(+, a)

#-

@time CUDA.@sync reduction1(+, a);
@time CUDA.@sync reduction2(+, a);

#=
# Reduction 3 : Sequential access

In both the above implementations our memory-access pattern is *strided* which is difficult to coalesce. We discussed *coalesced* memory access in the **Shared Memory** tutorial.

**TL;DR** When consecutive threads access consecutive locations in memory, the GPU combines several transactions into a fewer transactions which is called coalesced memory access. When memory accesses are not consecutive which happens when the locations are non-sequantial, sparse or misaligned the GPU hardware is unable to reduce the number of transactions. Since transactions are serviced sequentially there is a significant performance penalty for non-coalesced access.

To make use of sequantial access instead of `stride` iterating from 1 to `length ÷ 2` we can do it the other way around (`length ÷ 2`:1)

**NOTE**: The algorithm below assumes that the `blockDim` is a power of two. Transforming it to become friendly with non-power of two can be done as an exercise.
=#

function reduction3_kernel(op, a, sums)
    shmem = @cuDynamicSharedMem(eltype(a), (blockDim().x, ))
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index = threadIdx().x
    len = blockDim().x
    
    ## Adjust length for the last block
    if blockIdx().x == gridDim().x
        len = mod1(length(a), blockDim().x)
    end
    
    if tid <= length(a)
        @inbounds shmem[index] = a[tid]
    else
        return
    end
    sync_threads()

    stride = len ÷ 2
    while stride > 0
        if index <= stride && index + stride <= len
            shmem[index] = op(shmem[index], shmem[index + stride])
        end
        stride = stride ÷ 2
        sync_threads()
    end
    
    if index == 1
        @inbounds sums[blockIdx().x] = shmem[1]
    end
    
    return
end

#-

function reduction3(op, a::CuArray)
    threadsPerBlock = 1024
    len = length(a)
    
    sums = similar(a, cld(len, threadsPerBlock))
    
    blocks = cld(len, threadsPerBlock)
    shmem = sizeof(eltype(a))*threadsPerBlock
    @cuda shmem=shmem threads=threadsPerBlock blocks=blocks reduction3_kernel(op, a, sums)
    
    ## Recursively call reduction for larger arrays
    if length(sums) > 1
        return reduction3(op, sums)[1]
    end
    
    CUDA.@allowscalar return sums[1]
    return sums
end

#-

a = CUDA.ones(1024);
reduction3(+, a)

#-

a = CUDA.ones(Int, 1024 * 1024);
reduction3(+, a)

#-

CUDA.@time reduction2(+, a);
CUDA.@time reduction3(+, a);

#=
# Reduction 4 : Warp Shuffle

A powerful feature in modern GPUs is the ability to communicate within warps with the help of special instructions. Currently we transfer data with the help of shared memory which obviously requires `sync_threads()` and access to shared memory. Warp shuffle functions allow transferring memory within a warp without the use of shared memory also being much faster and not requiring any explicit barrier. The only drawback is that only the following primitive types are supported: `Int32, UInt32, Int64, UInt64, Float32, Float64` and any arbitrary source to destination lane mapping is not permitted.

There are four shuffle methods.
- `shfl_sync`
- `shfl_up_sync`
- `shfl_down_sync`
- `shfl_xor_sync`
=#

@doc shfl_sync

#=
`shfl_sync` acts as a broadcast transferring a lane's value to all other lane.
=#

function broadcast_gpu(lane)
    id = threadIdx().x
    val = id
    mask = typemax(UInt32) # 0xffffffff
    newval = shfl_sync(mask, val, lane)
    @cuprint("id: ", id, "\t value: ", val, "\t new value: ", newval, "\n")
    return
end

@cuda threads=32 blocks=1 broadcast_gpu(19)

#=
`shfl_up_sync` and `shfl_down_sync` copy the value from lane = current_lane ± delta. If lane is out of bounds from the warp then
=#

@doc shfl_down_sync

#-

function shfldown_gpu(delta)
    id = threadIdx().x
    val = id
    mask = typemax(UInt32) # 0xffffffff
    newval = shfl_down_sync(mask, val, delta)
    @cuprint("id: ", id, "\t old value: ", val, "\t new value: ", newval, "\n")
    return
end

@cuda threads=32 blocks=1 shfldown_gpu(2)
synchronize()

#=
We can use `shfl_down_sync` to reduce a warp much faster than shared memory.
=#

@inline function reducewarp(op, val, mask = typemax(UInt32))
    val = op(val, shfl_down_sync(mask, val, 1))
    val = op(val, shfl_down_sync(mask, val, 2))
    val = op(val, shfl_down_sync(mask, val, 4))
    val = op(val, shfl_down_sync(mask, val, 8))
    val = op(val, shfl_down_sync(mask, val, 16))
    return val
end

function reduction4_kernel(op, a, sums)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    lane_id = tid % 32
    warp_id = cld(tid, 32)
    
    ## exit 
    tid > length(a) && return
    
    ## set essential
    tid <= length(a) && (val = a[tid])
    
    val = reducewarp(op, val)
    lane_id == 1 && (sums[warp_id] = val)
    return
end

function reduction4(op, a::CuArray)
    threadsPerBlock = 1024
    len = length(a)
    
    sums = similar(a, cld(len, 32))
    
    blocks = cld(len, threadsPerBlock)
    shmem = sizeof(eltype(a))*threadsPerBlock
    @cuda threads=threadsPerBlock blocks=blocks reduction4_kernel(op, a, sums)
    
    ## Recursively call reduction for larger arrays
    if length(sums) > 1
        return reduction4(op, sums)
    end
    
    CUDA.@allowscalar return sums[1]
    return sums
end

#-

a = CUDA.ones(Int, 320_000)
reduction4(+, a)

#=
There is one big problem with our implementation, the input length is expected to be a multiple of 32. This problem can be solved either by defining a neutral element for `op` (`zero(eltype(a))` for `+`, `one(eltype(a))` for `*`) however this won't work with `xor`. Another is to force the last warp's computation via shared memory like earlier examples. 
We correct this by having only 1 thread work for the last warp; this is a simple solution that is inefficient but when the number of warps is large the performance hit shouldn't be too high.
=#

function reduction5_kernel(op, a, sums)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    lane_id = tid % 32
    warp_id = cld(tid, 32)
    
    ## exit non essential 
    tid > cld(length(a), 32)*32 && return
    
    ## set essential
    tid <= length(a) && (val = a[tid])
    
    ## Manage last warp
    if warp_id*32 > length(a)
        if lane_id == 1
            for i=(tid + 1):length(a)
                val = op(val, a[i])
            end
        end
    else
        val = reducewarp(op, val)  
    end
    
    lane_id == 1 && (sums[warp_id] = val)
    return
end

function reduction5(op, a::CuArray)
    threadsPerBlock = 1024
    len = length(a)
    
    sums = similar(a, cld(len, 32))
    
    blocks = cld(len, threadsPerBlock)
    shmem = sizeof(eltype(a))*threadsPerBlock
    @cuda threads=threadsPerBlock blocks=blocks reduction5_kernel(op, a, sums)
    
    ## Recursively call reduction for larger arrays
    if length(sums) > 1
        return reduction5(op, sums)
    end
    
    CUDA.@allowscalar return sums[1]
    return sums
end

#-

a = CUDA.ones(Int, 5_000_000)
reduction5(+, a)

#-

@time CUDA.@sync reduction3(+, a);
@time CUDA.@sync reduction5(+, a);
