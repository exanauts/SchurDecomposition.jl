#=
    Code taken from ProxAL.jl. Original file:
    https://github.com/exanauts/ProxAL.jl/blob/master/src/communication.jl

    All this file is subject to:

    MIT License

    Copyright (c) 2020 Anirudh Subramanyam and Youngdae Kim
=#
"""
    whoswork(blk, comm)

To which rank is block `blk` currently assigned to.

"""
function whoswork(blk, comm::MPI.Comm)
    blk % MPI.Comm_size(comm)
end

function whoswork(blk, comm::Nothing)
    error("Communicator Nothing should not ask this.")
    return nothing
end

"""
    is_my_work(blk, comm)

Returns a boolean whether the block `blk` is assigned to this rank.

"""
is_my_work(blk, comm::MPI.Comm) = (whoswork(blk, comm) == MPI.Comm_rank(comm))
is_my_work(blk, comm::Nothing) = true

"""
    comm_wait!(requests)

Wait until the communciation requests `requests` have been fulfilled.

"""
comm_wait!(requests::Vector{MPI.Request}) = MPI.Waitall!(requests)
comm_wait!(requests::Nothing) = nothing

"""
    comm_max(data, comm)

Collective to reduce and return the maximum over scalar `data`.

"""
comm_max(data::Float64, comm::MPI.Comm) = MPI.Allreduce(data, MPI.MAX, comm)
comm_max(data::Float64, comm::Nothing) = data

"""
    comm_sum(data::Float64, comm)

Collective to reduce and return the sum over scalar `data`.

"""
comm_sum(data::Float64, comm::MPI.Comm) = MPI.Allreduce(data, MPI.SUM, comm)
comm_sum(data::Float64, comm::Nothing) = data

"""
    comm_sum!(data, comm)

Collective to reduce the sum over array `data`.

"""
comm_sum!(data::AbstractArray, comm::MPI.Comm) = MPI.Allreduce!(data, MPI.SUM, comm)
comm_sum!(data::AbstractArray, comm::Nothing) = data
# Special handling for CUDA device
function comm_sum!(data::CUDA.CuArray, comm::MPI.Comm)
    CUDA.synchronize() # Ensure we synchronize all streams before Reduce
    MPI.Allreduce!(data, MPI.SUM, comm)
end

comm_rank(comm::MPI.Comm) = MPI.Comm_rank(comm)
comm_rank(comm::Nothing) = 0

comm_barrier(comm::MPI.Comm) = MPI.Barrier(comm)
comm_barrier(comm::Nothing) = nothing

