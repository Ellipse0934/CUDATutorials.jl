
using CUDA

function gpu_axpy!(A, X, Y)
    # set tid to thread rank
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


@show kernel_args = cudaconvert((α, gpu_v1, gpu_v3)) # Convert to GPU friendly types
@show kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
kernel = cufunction(gpu_axpy!, kernel_tt)
kernel_config = launch_configuration(kernel.fun, shmem = 0)


using Images

img = rand(Bool, 10, 10)
Gray.(img)


dims = (3000, 3000)

mset = Array{Bool, 2}(undef, dims)

function mandelbrot_cpu(mset::AbstractArray, dims, iterations)
    origin = CartesianIndex(div.(dims, (2, 2), RoundUp))
    for ind in CartesianIndices(mset)
        # Compute coordinates for true canvas
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


mandelbrot_cpu(mset, dims, 32)
Gray.(mset)


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


cmap = colormap("RdBu", 32 + 1)


map(x -> cmap[x + 1], mset_color)


mset_gpu = CuArray{UInt8}(undef, dims)
function mandelbrot_gpu(mset::AbstractArray, dims, iterations)
    ind = CartesianIndex((blockIdx().y - 1)*blockDim().y + threadIdx().y,
                        (blockIdx().x - 1)*blockDim().x + threadIdx().x)
    # Check if index is valid, if not then exit
    !(ind in CartesianIndices(dims)) && return
    origin = CartesianIndex(div.(dims, (2, 2), RoundUp))
    
    # Scale the 3000x3000 image to -1.5 to 1.5
    coordinates = Tuple(ind - origin) ./ 1000.
    c = ComplexF32(coordinates[1]im + coordinates[2]) # x + yi
    mset[ind] = mandelbrot(c, iterations)
    return
end


blkdim = (16, 16)
@cuda blocks=cld.(dims, blkdim) threads=blkdim mandelbrot_gpu(mset_gpu, dims, 32)


# copy back to host and display the same image
map(x -> cmap[x + 1], Array(mset_gpu))

