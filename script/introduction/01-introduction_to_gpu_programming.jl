
using CUDA
CUDA.functional()


function axpy!(A, X, Y) 
    for i in eachindex(Y)
        @inbounds Y[i] = A * X[i] + Y[i]
    end
end

N = 2^29
v1 = rand(Float32, N)
v2 = rand(Float32, N)
v2_copy = copy(v2) # maintain a copy of the original
α = rand()

axpy!(α, v1, v2)


using Base.Threads

println("Number of CPU threads = ", nthreads())

# pseudocode for parallel saxpy
function parallel_axpy!(A, X, Y)
    len = cld(length(X), nthreads())

    # Launch threads = nthreads()
    Threads.@threads for i in 1:nthreads()
        # set id to thread rank/id
        tid = threadid()
        low = 1 + (tid - 1)*len
        high = min(length(X), len * tid) # The last segment might have lesser elements than len

        # Broadcast syntax, views used to avoid copying
        view(Y, low:high) .+= A.*view(X, low:high)
    end
    return
end

v4 = copy(v2_copy)
parallel_axpy!(α, v1, v4)

@show v2 == v4


function gpu_axpy!(A, X, Y) 
    # set tid to thread rank
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid > length(Y) && return 
    @inbounds Y[tid] = A*X[tid] + Y[tid]
    return
end

# Transfer array to GPU memory
gpu_v1 = CuArray(v1)
gpu_v2 = CuArray(v2_copy)

numthreads = 256
numblocks = cld(N, numthreads)

@show numthreads
@show numblocks

# Launch the gpu_axpy! on the GPU
@cuda threads=numthreads blocks=numblocks gpu_axpy!(α, gpu_v1, gpu_v2)

# Copy back to RAM
v4 = Array(gpu_v2)

# Verify that the answers are the same
@show v2 == v4


arr = CUDA.rand(10);
arr[1]


CUDA.allowscalar(false)
arr[1]


CUDA.@allowscalar arr[1]


@show CUDA.available_memory 
@show CUDA.total_memory();


@device_code_ptx @cuda threads=numthreads blocks=numblocks gpu_axpy!(α, gpu_v1, gpu_v2)


@time axpy!(α, v1, v2)
@time parallel_axpy!(α, v1, v2)
@time @cuda threads=numthreads blocks=numblocks gpu_axpy!(α, gpu_v1, gpu_v2)
sleep(0.1) # complete for previous function to finish
@time CUDA.@sync @cuda threads=numthreads blocks=numblocks gpu_axpy!(α, gpu_v1, gpu_v2)
CUDA.@time @cuda threads=numthreads blocks=numblocks gpu_axpy!(α, gpu_v1, gpu_v2)

# TODO: Add a scatter plot of time vs array size and link to code snippet

