module LibPardiso

const libpardiso = "/home/fpacaud/lib/panua-pardiso-20230908-linux/lib/libpardiso.so"

struct doublecomplex
    re::Cdouble
    i::Cdouble
end

function pardisoinit(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:pardisoinit, libpardiso), Cvoid, (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function pardiso(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17)
    ccall((:pardiso, libpardiso), Cvoid, (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cdouble}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17)
end

function pardiso_chkmatrix(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:pardiso_chkmatrix, libpardiso), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function pardiso_chkvec(arg1, arg2, arg3, arg4)
    ccall((:pardiso_chkvec, libpardiso), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}), arg1, arg2, arg3, arg4)
end

function pardiso_printstats(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:pardiso_printstats, libpardiso), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function pardisoinit_z(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:pardisoinit_z, libpardiso), Cvoid, (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function pardiso_z(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17)
    ccall((:pardiso_z, libpardiso), Cvoid, (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cdouble}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17)
end

function pardiso_chkmatrix_z(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:pardiso_chkmatrix_z, libpardiso), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function pardiso_chkvec_z(arg1, arg2, arg3, arg4)
    ccall((:pardiso_chkvec_z, libpardiso), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}), arg1, arg2, arg3, arg4)
end

function pardiso_printstats_z(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:pardiso_printstats_z, libpardiso), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function pardiso_get_schur_z(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:pardiso_get_schur_z, libpardiso), Cvoid, (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

function pardisoinit_d(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:pardisoinit_d, libpardiso), Cvoid, (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function pardiso_d(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17)
    ccall((:pardiso_d, libpardiso), Cvoid, (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cdouble}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17)
end

function pardiso_chkmatrix_d(arg1, arg2, arg3, arg4, arg5, arg6)
    ccall((:pardiso_chkmatrix_d, libpardiso), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6)
end

function pardiso_chkvec_d(arg1, arg2, arg3, arg4)
    ccall((:pardiso_chkvec_d, libpardiso), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}), arg1, arg2, arg3, arg4)
end

function pardiso_printstats_d(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
    ccall((:pardiso_printstats_d, libpardiso), Cvoid, (Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
end

function pardiso_get_schur_d(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    ccall((:pardiso_get_schur_d, libpardiso), Cvoid, (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}), arg1, arg2, arg3, arg4, arg5, arg6, arg7)
end

end # module
