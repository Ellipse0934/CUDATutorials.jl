
using CUDA, BenchmarkTools

function reverse(input, output = similar(input))
    len = length(input)
    for i = 1:cld(len,2)
        output[i], output[len - i + 1] = input[len - i + 1], input[i]
    end
    output
end


reverse([1, 2, 3, 4, 5])


function gpu_reverse(input, output)
    tid = threadIdx().x
    len = length(input)
    if tid <= cld(len, 2)
        output[tid], output[len - tid + 1] = input[len - tid + 1], input[tid]
    end
    return
end


A = CuArray(collect(1:5))
B = similar(A)
@cuda blocks=1 threads=length(A) gpu_reverse(A, B)
B


?@cuStaticSharedMem


function gpu_stshmemreverse(input, output)
    # Maximum size of array is 64
    shmem = @cuStaticSharedMem(eltype(output), 64)
    tid = threadIdx().x
    len = length(input)
    shmem[tid] = input[len - tid + 1]
    output[tid] = shmem[tid]
    return
end


A = CuArray(collect(1:32))
B = similar(A)
@cuda blocks=1 threads=length(A) gpu_stshmemreverse(A, B)
print(B)


?@cuDynamicSharedMem


function gpu_dyshmemreverse(input, output)
    shmem = @cuDynamicSharedMem(eltype(output), (length(output),))
    tid = threadIdx().x
    len = length(input)
    shmem[tid] = input[len - tid + 1]
    output[tid] = shmem[tid]
    return
end


C = CuArray(collect(1:32))
D = similar(C)
@cuda blocks=1 threads=length(C) shmem=length(C) gpu_dyshmemreverse(C, D)
print(D)


A = reshape(1:9, (3, 3))


A'


A'' # (A')'


# CPU implementation
function cpu_transpose(input, output = similar(input, (size(input, 2), size(input, 1))))
    # the dimensions of the resultant matrix are reversed
    for index = CartesianIndices(input)
            output[index[2], index[1]] = input[index]
    end
    output
end


A = reshape(1:20, (4, 5))


cpu_transpose(A)


A = CuArray(reshape(1.:9, (3, 3)))

println("A => ", pointer(A))
CUDA.@allowscalar begin
    for i in eachindex(A)
        println(i, " ", A[i], " ", pointer(A, i))
    end
end


# To index our 2-D array we will split the input into tiles of 32x32 elements. 
# Each thread block will launch with 32x8 = 256 threads 
# Each thread will work on 4 elements.
const TILE_DIM = 32

function gpu_transpose_kernel(input, output)
    tile_index = ((blockIdx().y, blockIdx().x) .- 1) .* TILE_DIM
    
    # each thread manages 4 rows (8x4 = 32)
    for i in 1:4
        thread_index = (threadIdx().y + (i - 1)*8, threadIdx().x)
        index = CartesianIndex(tile_index .+ thread_index)
        (index[1] > size(input, 1) || index[2] > size(input, 2)) && continue
        @inbounds output[index] = input[index[2], index[1]]
    end

    return
end


function gpu_transpose(input, output = similar(input, (size(input, 2), size(input, 1))))
    threads = (32, 8)
    blocks = cld.(size(input), (32, 32))
    @cuda blocks=blocks threads=threads gpu_transpose_kernel(input, output)
    output
end


A = CuArray(reshape(1f0:1089, 33, 33))


gpu_transpose(A)


A = CUDA.rand(10000, 10000)
B = similar(A)
@benchmark CUDA.@sync gpu_transpose(A, B)


@benchmark CUDA.@sync B .= A


#=
# Not sure if this should be included, custom kernel does happen to be faster
# than the broadcast copy.

function gpu_copy_kernel(input, output)
    x_index = (blockIdx().x - 1)*TILE_DIM + threadIdx().y
    y_index = (blockIdx().y - 1)*TILE_DIM + threadIdx().x
    
    for i in 1:4 # each thread needs to manage 4 rows (8x4 = 32)
        index = CartesianIndex(y_index , x_index + (i - 1)*8)
        (index[1] > size(input, 1) || index[2] > size(input, 2)) && continue
        @inbounds output[index] = input[index]
    end
    
    return
end

function gpu_copy(input, output = similar(input, size(input)))
    threads = (32, 8)
    blocks = cld.(size(input), (32, 32))
    @cuda blocks=blocks threads=threads gpu_copy_kernel(input, output)
    output
end
A = CUDA.rand(12000, 12000)
B = similar(A)
@benchmark CUDA.@sync gpu_copy(A, B)
=#


function gpu_transpose_kernel2(input, output)
    # Declare shared memory
    shared = @cuStaticSharedMem(eltype(input), (TILE_DIM, TILE_DIM))
    
    # Modify thread index so threadIdx().x dominates the column
    block_index = ((blockIdx().y, blockIdx().x) .- 1) .* TILE_DIM
    
    for i in 1:4
        thread_index = (threadIdx().x, threadIdx().y + (i - 1)*8)
        index = CartesianIndex(block_index .+ thread_index)

        (index[1] > size(input, 1) || index[2] > size(input, 2)) && continue
        @inbounds shared[thread_index[2], thread_index[1]] = input[index]
    end
    
    # Barrier to ensure all threads have completed writing to shared memory
    sync_threads()
    
    # swap tile index
    block_index = ((blockIdx().x, blockIdx().y) .- 1) .* TILE_DIM
    
    for i in 1:4 
        thread_index = (threadIdx().x, threadIdx().y + (i - 1)*8)
        index = CartesianIndex(block_index .+ thread_index)
        
        (index[1] > size(output, 1) || index[2] > size(output, 2)) && continue
        @inbounds output[index] = shared[thread_index...]
    end
    return
end

function gpu_transpose_shmem(input, output = similar(input, (size(input, 2), size(input, 1))))
    threads = (32, 8)
    blocks = cld.(size(input), (32, 32))
    @cuda blocks=blocks threads=threads gpu_transpose_kernel2(input, output)
    output
end


A = CuArray(reshape(1f0:1089, (33, 33)))
B = similar(A)
gpu_transpose_shmem(A, B)


A = CUDA.rand(10000, 10000)
B = similar(A)
@benchmark CUDA.@sync gpu_transpose_shmem(A, B)


@benchmark CUDA.@sync B .= A

# TODO: Doesn't look like an inspiring case. We should investigate why the broadcast is slower.
# ofc make it faster so that there is an inspiring claim of how there is more room 
# for performance. Making "bank conflicts" the next natural topic.


function gpu_transpose_kernel3(input, output)
    # Declare shared memory
    shared = @cuStaticSharedMem(eltype(input), (TILE_DIM + 1, TILE_DIM))
    
    # Modify thread index so threadIdx().x dominates the column
    block_index = ((blockIdx().y, blockIdx().x) .- 1) .* TILE_DIM
    
    for i in 1:4
        thread_index = (threadIdx().x, threadIdx().y + (i - 1)*8)
        index = CartesianIndex(block_index .+ thread_index)

        (index[1] > size(input, 1) || index[2] > size(input, 2)) && continue
        @inbounds shared[thread_index[2], thread_index[1]] = input[index]
    end
    
    # Barrier to ensure all threads have completed writing to shared memory
    sync_threads()
    
    # swap tile index
    block_index = ((blockIdx().x, blockIdx().y) .- 1) .* TILE_DIM
    
    for i in 1:4 
        thread_index = (threadIdx().x, threadIdx().y + (i - 1)*8)
        index = CartesianIndex(block_index .+ thread_index)
        
        (index[1] > size(output, 1) || index[2] > size(output, 2)) && continue
        @inbounds output[index] = shared[thread_index...]
    end
    return
end

function gpu_transpose_noconf(input, output = similar(input, (size(input, 2), size(input, 1))))
    threads = (32, 8)
    blocks = cld.(size(input), (32, 32))
    @cuda blocks=blocks threads=threads gpu_transpose_kernel3(input, output)
    output
end


A = CUDA.rand(10000, 10000)
B = similar(A)
@benchmark CUDA.@sync gpu_transpose_noconf(A, B)


A = CUDA.rand(10000, 10000)
B = similar(A)
@benchmark CUDA.@sync gpu_transpose_shmem(A, B)

