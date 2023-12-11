module SchurDecomposition

import MPI

import LinearAlgebra: mul!, ldiv!, axpy!, norm, Symmetric, I
import SparseArrays: SparseMatrixCSC, sparse, spzeros, nnz, dropzeros!
import MadNLP
import MadNLPHSL
import NLPModels
import ExaPF
import Argos

import CUDA
import KernelAbstractions as KA

const MPI_ROOT = 0

# Load library
include(joinpath(@__DIR__, "..", "lib", "libpardiso.jl"))

include("utils.jl")
include("communication.jl")
include("blockopfmodel.jl")
include("schur_decomposition.jl")

include("schur_solver.jl")
include("parallel_kkt.jl")

end # module
