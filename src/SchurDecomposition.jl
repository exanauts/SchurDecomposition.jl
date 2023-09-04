module SchurDecomposition

import MPI

import LinearAlgebra: mul!, ldiv!, axpy!
import SparseArrays: SparseMatrixCSC, sparse
import MadNLP
import MadNLPHSL
import NLPModels
import ExaPF
import Argos

import CUDA
import KernelAbstractions as KA

const MPI_ROOT = 0

include("utils.jl")
include("communication.jl")
include("blockopfmodel.jl")
include("schur_decomposition.jl")
include("parallel_kkt.jl")

end # module
