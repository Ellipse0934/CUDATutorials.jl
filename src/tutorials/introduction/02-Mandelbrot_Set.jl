#=
# Mandelbrot Set
In the last tutorial we got a brief overview of the architecture of a GPU. In this tutorial We will elborate a bit more on Grid configuration and compute the mandelbrot set on the GPU.


## Grid configuration

When we launch a CUDA kernel we use the syntax `@cuda blocks=x threads=y f(a, b, c...)` where `f` is the function and `a, b, c...` are its arguments.

The `blocks` and `threads` options to `@cuda` are used to specify the **grid** configuration of the kernel which is being launched. 

`blocks` specifies the dimension and the size of the grid of blocks. This can be one, two or three dimensional. The reason for having multiple dimensions is to make it easier to express algorithms which index over two or three dimensional spaces. 

`threads` specifies the *cooperative thread array* (CTA). Threads in the same CTA have access to better coordination and communication utilities. CTA's can also two or three dimensional for convenient indexing.

There are restrictions related to the grid given below.
* Maximum x-dimension of a grid of thread blocks : $2^{31} - 1$
* Maximum y-, or z-dimension of a grid of thread blocks : $65535 (2^{16} - 1)$
* Maximum x- or y-dimension of a CTA: $1024$
* Maximum z-dimension of a CTA: $64$
* Maximum number of threads per block: $1024$ 

Now let's go back to our SAXPY example and verify the flexibility in choosing grid configurations.
=#

using CUDA, BenchmarkTools

function gpu_axpy!(A, X, Y)
    ## set tid to thread rank
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid > length(X) && return 
    Y[tid] = A*X[tid] + Y[tid]
    return
end

N = 2^20
gpu_v1 = CUDA.rand(N)
gpu_v2 = CUDA.rand(N)
gpu_v3 = copy(gpu_v2)

α = 0.48
gpu_v2 .+= α * gpu_v1

function verify_grid(args, result, numthreads, numblocks = cld(N, numthreads))
    u = copy(args[3])

    @cuda threads=numthreads blocks=numblocks gpu_axpy!(args...)
    println("Explicit kernel launch with threads=$numthreads and blocks=$numblocks is correct: "
    ,result == args[3])

    args[3] = u
    return 
end

args = [α, gpu_v1, gpu_v3]

verify_grid(args, gpu_v2, 1024)
verify_grid(args, gpu_v2, 1)
verify_grid(args, gpu_v2, 33)

#=
## Occupancy

The above exercise shows the flexibility in deciding grid configuration. However this raises an important question of how the configuration affects performance. The best way to determine is to actually measure what works best in a given scenario however that may prove to be cumbersome for most workflows. Expecially while developing an application it may not be worth our time to find the optimal configuration.

While each SM(streaming multiprocessor) might be executing 100's of threads from the perspective of the programmer, the GPU Hardware deals with `warps`. Each warp is a set of fixed number of threads(32 on NVIDIA hardware). Scheduling and issuing instructions is done at a per warp basis rather than a per thread basis. There is atleast one *warp scheduler* inside each SM whose job it is to keep the SM as busy as possible. Switching between warps(context switch) is extremely fast and essential to hide latencies such as global memory access. Whenever a warp stalls another warp is immediately switched to.

Coming back to out original question of how to determine the optimal grid configuration. One possible solution is to use the occupancy heurestic. $\mbox{occupancy} = \frac{\mbox{active warps}}{\mbox{maximum number of active warps}}$.
Since each SM has a finite amount of resources. As the number of resources per thread increases, fewer of them can be concurrently executed. Occupancy can limited by register usage, shared memory and block size.

The `launch_configuration` function analyses the kernel's resource usage and suggests a configuration that maximises occupancy.
=#

@show kernel_args = cudaconvert((α, gpu_v1, gpu_v3)) # Convert to GPU friendly types
@show kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
kernel = cufunction(gpu_axpy!, kernel_tt)
kernel_config = launch_configuration(kernel.fun, shmem = 0)

#=
For our example the configurator returns 20 blocks and 1024 threads. The occupancy API does not understand what our kernel is doing, it can only see the input types and the function definition. It's our job to figure out if the suggested configuration will work or not. It's best to keep the suggested block size in account while deciding the launch config.

For this example it's perhaps best to set the block size to 1024 and determine the grid size based on that.

# Mandelbrot Set

A popular example of mathematical visualization is the Mandelbrot set. Mathematically it is defined as the set of [complex numbers](https://simple.wikipedia.org/wiki/Complex_number) $c$ such $f_c(z) = z^2 + c$ is bounded.

In other words:
* ‎‎$Z_0 = 0$‎
* $Z_{n + 1} = {Z_n}^2 + c$
* $c$ is in the mandelbrot set if the value of $Z_{n}$ is bounded.

It can be mathematically shown $|Z_n| \leq 2.0 $ for bounded points.
=#

using Images

img = rand(Bool, 10, 10)
Gray.(img)

#-

dims = (3000, 3000)

mset = Array{Bool, 2}(undef, dims)

function mandelbrot_cpu(mset::AbstractArray, dims, iterations)
    origin = CartesianIndex(div.(dims, (2, 2), RoundUp))
    for ind in CartesianIndices(mset)
        ## Compute coordinates for true canvas
        coordinates = Tuple(ind - origin) ./ 1000.
        c = ComplexF32(coordinates[1]im + coordinates[2])
        mset[ind] = mandelbrot(c, iterations)
    end
end

function mandelbrot(c, iterations)
    z = ComplexF32(0, 0)
    for i in 1:iterations
        z = z^2 + c
        abs(z) > 2.0 && return false
    end
    return true
end

#-

mandelbrot_cpu(mset, dims, 32)
Gray.(mset)

# This black and white image is no fun. To add color let's map a color to the number iterations it took $z$ to become greater than two.

mset_color = Array{UInt8}(undef, dims)
function mandelbrot(c, iterations)
    z = ComplexF32(0, 0)
    for i in 1:iterations
        z = z^2 + c
        abs2(z) > 4.0 && return i % UInt8
    end
    return zero(UInt8) 
end
mandelbrot_cpu(mset_color, dims, 32)

#-

cmap = colormap("RdBu", 32 + 1)

#-

map(x -> cmap[x + 1], mset_color)

#=
Our task is to move this computation to the GPU. The tricky part with moving to the GPU is that the idexing gets tricky. We can use 1-D indexing then figure out inside the kernel what our 2-D index is or use 2-D index from the get go. 
=#

mset_gpu = CuArray{UInt8}(undef, dims)
function mandelbrot_gpu(mset::AbstractArray, dims, iterations)
    ind = CartesianIndex((blockIdx().y - 1)*blockDim().y + threadIdx().y,
                        (blockIdx().x - 1)*blockDim().x + threadIdx().x)
    ## Check if index is valid, if not then exit
    !(ind in CartesianIndices(dims)) && return
    origin = CartesianIndex(div.(dims, (2, 2), RoundUp))
    
    ## Scale the 3000x3000 image to -1.5 to 1.5
    coordinates = Tuple(ind - origin) ./ 1000.
    c = ComplexF32(coordinates[1]im + coordinates[2]) # x + yi
    mset[ind] = mandelbrot(c, iterations)
    return
end

#-

blkdim = (16, 16)
@cuda blocks=cld.(dims, blkdim) threads=blkdim mandelbrot_gpu(mset_gpu, dims, 32)

#-

## copy back to host and display the same image
map(x -> cmap[x + 1], Array(mset_gpu))

#=
## Thread Divergence

We mentioned in the last tutorial that threads in a warp execute the same instruction. This was not the entire picture as you can guess from the mandelbrot set example. Inside the `mandelbrot` inner loop we have consecutive threads exiting at different points during iteration.


```julia
    for i in 1:iterations
        z = z^2 + c
        abs2(z) > 4.0 && return i % UInt8
    end
```

When threads of the same warp are following different execution paths we call it thread divergence. Even when 1 thread out of 32 follows takes a different branch there is thread divergence.

For example

```julia
if threadIdx().x % 2 == 0
    # Do something
else
    # Do something else
end
```

In the above example threads with an even index will follow the first path and the odd indexed ones will follow the second path. Inside the GPU when a branching condition is evaluated a 32-bit mask is generated for that warp (1-bit for each lane). All lanes whose corresponding mask bit is true will be active and the remaining lanes will be idle. When execution reaches a convergence point, the mask is inverted so that all lanes which were idle become active and vice versa. This kind of an IF-ELSE branch has 50% efficiency. Even if a single thread had diverged (say condition was `threadIdx().x % 32 == 0`) we would still be at 50% efficiency. However, nesting IF-ELSE will further reduce efficiency.

In our mandelbrot example while we definitely had a lot of thread divergence it was still beneficial because threads in a warp represented physically close pixels which diverge less. 

Also, note that thread divergence refers to intra-warp divergence rather than inter-warp divergence which does not matter to performance. Because doing work efficiently at the warp level is extremely important for performance we will consider it while writing algorithms. In addition there are a number of functions that work at the warp-level such as `sync_warp()`and `shfl_up_sync()`. We will explore these in the `reduction` and `prefix scan` tutorials.
=#