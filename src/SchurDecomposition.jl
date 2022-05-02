module SchurDecomposition

import MPI

import LinearAlgebra: mul!, ldiv!, axpy!
import MadNLP
import NLPModels
import Argos

import KernelAbstractions as KA

const MPI_ROOT = 0

include("communication.jl")
include("blockopfmodel.jl")
include("schur_decomposition.jl")

end # module
