
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
    @inbounds for i in 1:n, j in 1:i
        val = (A[i, j] + A[j, i]) / 2.0
        A[i, j] = val
        A[j, i] = val
    end
end


struct AugmentedKKTMatrix{T, VT, MT}
    # Sparse structure
    hess::VT
    jac_callback::VT
    jac::VT
    # Augmented system
    aug_raw::MadNLP.SparseMatrixCOO{T,Int32,VT, Vector{Int32}}
    aug_com::MT
    aug_csc_map::Union{Nothing, Vector{Int}}
    # Hessian
    hess_raw::MadNLP.SparseMatrixCOO{T,Int32,VT, Vector{Int32}}
    hess_com::MT
    hess_csc_map::Union{Nothing, Vector{Int}}
    # Jacobian
    jac_raw::MadNLP.SparseMatrixCOO{T,Int32,VT, Vector{Int32}}
    jac_com::MT
    jac_csc_map::Union{Nothing, Vector{Int}}
    pr_diag::VT
    du_diag::VT
end

function AugmentedKKTMatrix(
    cb::MadNLP.SparseCallback{T,VT},
    ind_cons,
) where {T,VT}
    n_slack = length(ind_cons.ind_ineq)
    # Deduce KKT size.
    n = cb.nvar
    m = cb.ncon
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)
    ind_ineq = ind_cons.ind_ineq

    # Jacobian
    jac_sparsity_I = MadNLP.create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = MadNLP.create_array(cb, Int32, cb.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    # Hessian
    hess_sparsity_I, hess_sparsity_J = MadNLP.build_hessian_structure(cb, MadNLP.ExactHessian)
    MadNLP.force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack

    aug_vec_length = n_tot+m
    aug_mat_length = n_tot+m+n_hess+n_jac+n_slack

    I = MadNLP.create_array(cb, Int32, aug_mat_length)
    J = MadNLP.create_array(cb, Int32, aug_mat_length)
    V = VT(undef, aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    offset = n_tot+n_jac+n_slack+n_hess+m

    I[1:n_tot] .= 1:n_tot
    I[n_tot+1:n_tot+n_hess] .= hess_sparsity_I
    I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I.+n_tot)
    I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
    I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    J[1:n_tot] .= 1:n_tot
    J[n_tot+1:n_tot+n_hess] .= hess_sparsity_J
    J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    pr_diag = MadNLP._madnlp_unsafe_wrap(V, n_tot)
    du_diag = MadNLP._madnlp_unsafe_wrap(V, m, n_jac+n_slack+n_hess+n_tot+1)

    hess = MadNLP._madnlp_unsafe_wrap(V, n_hess, n_tot+1)
    jac = MadNLP._madnlp_unsafe_wrap(V, n_jac+n_slack, n_hess+n_tot+1)
    jac_callback = MadNLP._madnlp_unsafe_wrap(V, n_jac, n_hess+n_tot+1)

    aug_raw = MadNLP.SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = MadNLP.SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )
    hess_raw = MadNLP.SparseMatrixCOO(
        n_tot, n_tot,
        hess_sparsity_I,
        hess_sparsity_J,
        hess,
    )

    aug_com, aug_csc_map = MadNLP.coo_to_csc(aug_raw)
    jac_com, jac_csc_map = MadNLP.coo_to_csc(jac_raw)
    hess_com, hess_csc_map = MadNLP.coo_to_csc(hess_raw)
    return AugmentedKKTMatrix(
        hess,
        jac_callback,
        jac,
        aug_raw,
        aug_com,
        aug_csc_map,
        hess_raw,
        hess_com,
        hess_csc_map,
        jac_raw,
        jac_com,
        jac_csc_map,
        pr_diag,
        du_diag,
    )
end
