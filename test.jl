using Libdl
Libdl.dlopen("/opt/cray/pe/mpich/8.1.16/gtl/lib/libmpi_gtl_cuda.so.0.0.0", RTLD_GLOBAL)
using CUDA
using MPI


function init!(a::AbstractVector, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    if rank == 0
        fill!(a, 1)
    elseif rank == 1
        fill!(a, 0)
    else
        error("Wrong rank $rank")
    end
end

function check!(a::AbstractVector, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    if rank == 0
        if !(all(a .== 1))
            error("Wrong values on rank $rank with type $(typeof(a))")
        else
            println("Correct")
        end
    elseif rank == 1
        if !(all(a .== 1))
            error("Wrong values on rank $rank with type $(typeof(a))")
        else
            println("Correct")
        end
    else
        error("Wrong rank $rank")
    end
end

function sendrecv!(a::AbstractVector, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    if rank == 0
        MPI.Send(a, 1, 0, comm)
    elseif rank == 1
        MPI.Recv!(a, 0, 0, comm)
    else
        error("Wrong rank $rank")
    end
end

MPI.Init()
# Run with two ranks
println("MPI.has_cuda: $(MPI.has_cuda())")
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)
println("Size: $size")
@assert size == 2

a = Vector{Int64}(undef, 1)
init!(a, comm)
sendrecv!(a, comm)
check!(a, comm)

cua = CuVector{Int64}(undef, 1)
init!(cua, comm)
sendrecv!(cua, comm)
check!(cua, comm)
MPI.Finalize()
