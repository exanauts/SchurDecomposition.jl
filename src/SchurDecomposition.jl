module SchurDecomposition

import MPI

import MadNLP
import NLPModels
import Argos

const MPI_ROOT = 0

include("communication.jl")
include("blockopfmodel.jl")

end # module
