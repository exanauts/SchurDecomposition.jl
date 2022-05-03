
const DATA = "../matpower/data/" #joinpath(artifact"ExaData", "ExaData")
const DEMANDS = joinpath(artifact"ExaData", "ExaData", "mp_demand")

casename = "case1354pegase"
nscen = 12

## Options
scaling = true
max_iter = 100
