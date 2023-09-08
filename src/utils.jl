
# Split KKT system into Ki, K0 and Bi
# W is a lower triangular matrix
function split_kkt_system(W::SparseMatrixCSC, nx::Int, nu::Int)
    N = size(W, 1)

    nnzK0 = 0
    nnzKi = 0
    nnzBi = 0

    # Count elements in each matrix
    for j in 1:N
        for c in W.colptr[j]:W.colptr[j+1]-1
            i = W.rowval[c]
            if (nx + 1 <= i <= nx + nu) && (nx + 1 <= j <= nx + nu)
                nnzK0 += 1
            elseif (nx + 1 <= i <= nx + nu) && (j <= nx)
                nnzBi += 1
            elseif (nx + 1 <= j <= nx + nu)
                nnzBi += 1
            else
                nnzKi += 1
            end
        end
    end

    # Build W2
    mapK0 = zeros(Int32, nnzK0)
    i_K0 = zeros(Int32, nnzK0)
    j_K0 = zeros(Int32, nnzK0)

    mapKi = zeros(Int32, nnzKi)
    i_Ki = zeros(Int32, nnzKi)
    j_Ki = zeros(Int32, nnzKi)

    mapBi = zeros(Int32, nnzBi)
    i_Bi = zeros(Int32, nnzBi)
    j_Bi = zeros(Int32, nnzBi)

    k = 1
    lK0, lKi, lBi = 1, 1, 1

    for j in 1:N
        for c in W.colptr[j]:W.colptr[j+1]-1
            i = W.rowval[c]
            # Wuu
            if (nx + 1 <= i <= nx + nu) && (nx + 1 <= j <= nx + nu)
                i_K0[lK0] = i - nx
                j_K0[lK0] = j - nx
                mapK0[lK0] = k
                lK0 += 1
            # Wxu
            elseif (nx + 1 <= i <= nx + nu) && (j <= nx)
                # Transpose Wxu directly
                i_Bi[lBi] = j
                j_Bi[lBi] = i - nx
                mapBi[lBi] = k
                lBi += 1
            # Gu & Hu
            elseif (nx + 1 <= j <= nx + nu)
                i_Bi[lBi] = i - nu
                j_Bi[lBi] = j - nx
                mapBi[lBi] = k
                lBi += 1
            else
                i_Ki[lKi] = (i <= nx) ? i : i - nu
                j_Ki[lKi] = (j <= nx) ? j : j - nu
                mapKi[lKi] = k
                lKi += 1
            end
            k += 1
        end
    end

    return (
        i_Ki=i_Ki,
        j_Ki=j_Ki,
        mapKi=mapKi,
        i_K0=i_K0,
        j_K0=j_K0,
        mapK0=mapK0,
        i_Bi=i_Bi,
        j_Bi=j_Bi,
        mapBi=mapBi,
    )
end

function symmetrize!(A::AbstractMatrix)
    n = size(A, 1)
    for i in 1:n, j in 1:i
        val = (A[i, j] + A[j, i]) / 2.0
        A[i, j] = val
        A[j, i] = val
    end
end

#=
    Multi-RHS
=#
struct MultipleRHSSolver{T}
    solver::MadNLPHSL.Ma57Solver
    V::Matrix{T}
    lwork::Vector{T}
end

function MultipleRHSSolver(solver::MadNLPHSL.Ma57Solver, nrhs)
    n = size(solver.csc, 1)
    # Build buffers
    V = zeros(n, nrhs)
    lwork = zeros(n * nrhs)
    return MultipleRHSSolver(solver, V, lwork)
end

function _ma57cd!(
    job::Cint,n::Cint,fact::Vector{Float64},lfact::Cint,
    ifact::Vector{Cint},lifact::Cint,nrhs::Cint,rhs::Array{Float64},
    lrhs::Cint,work::Vector{Float64},lwork::Cint,iwork::Vector{Cint},
    icntl::Vector{Cint},info::Vector{Cint},
)
    ccall(
        ("ma57cd_", MadNLPHSL.libhsl),
        Nothing,
        (Ref{Cint},Ref{Cint},Ptr{Float64},Ref{Cint},
            Ptr{Cint},Ref{Cint},Ref{Cint},Ptr{Float64},
            Ref{Cint},Ptr{Float64},Ref{Cint},Ptr{Cint},
            Ptr{Cint},Ptr{Cint}),
        job,n,fact,lfact,ifact,lifact,nrhs,rhs,lrhs,work,lwork,iwork,icntl,info,
    )
end

function solve!(rsol::MultipleRHSSolver, X::AbstractMatrix)
    M = rsol.solver
    n, m = size(X)
    @assert m == size(rsol.V, 2)
    nrhs = m

    rsol.V .= X
    _ma57cd!(
        one(Int32),Int32(M.csc.n),M.fact,M.lfact,M.ifact,
        M.lifact,Int32(nrhs),rsol.V,Int32(n),rsol.lwork,Int32(n*nrhs),M.iwork,M.icntl,M.info,
    )
    M.info[1] < 0 && throw(MadNLP.SolveException())
    return rsol.V
end
    # for i in 1:kkt.n_coupling
    #     y = view(kkt.S, :, i)
    #     rhs .= kkt.Bi[:, i]
    #     MadNLP.solve!(kkt.sparse_solver, rhs)
    #     mul!(y, kkt.Bi', rhs, -1.0, 1.0)
    # end

