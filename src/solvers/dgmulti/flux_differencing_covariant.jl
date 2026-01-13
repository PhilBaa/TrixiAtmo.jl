function Trixi.create_cache(mesh::DGMultiMesh, equations::AbstractCovariantEquations, dg::Trixi.DGMultiFluxDiffSBP,
                            metric_terms, auxiliary_field, RealT,
                            uEltype)
    rd = dg.basis
    md = mesh.md

    # for use with flux differencing schemes
    Qrst_skew = Trixi.compute_flux_differencing_SBP_matrices(dg)

    # Todo: DGMulti. Factor common storage into a struct (MeshDataCache?) for reuse across solvers?
    # storage for volume quadrature values, face quadrature values, flux values
    nvars = nvariables(equations)
    naux = n_aux_node_vars(equations)
    u_values = Trixi.allocate_nested_array(uEltype, nvars, size(md.xq), dg)
    u_face_values = Trixi.allocate_nested_array(uEltype, nvars, size(md.xf), dg)
    flux_face_values = Trixi.allocate_nested_array(uEltype, nvars, size(md.xf), dg)
    aux_values = Trixi.allocate_nested_array(uEltype, naux, size(md.x), dg)
    aux_values = StructArray(aux_values) # use StructArray for better memory access in flux differencing
    aux_quad_values = Trixi.allocate_nested_array(uEltype, naux, size(md.xq), dg)
    aux_quad_values = StructArray(aux_quad_values)
    aux_face_values = Trixi.allocate_nested_array(uEltype, naux, size(md.xf), dg)
    aux_face_values = StructArray(aux_face_values)

    init_auxiliary_node_variables!(aux_values, mesh, equations, dg, auxiliary_field)
    aux_quad_values .= aux_values
    Trixi.apply_to_each_field(Trixi.mul_by!(rd.Vf), aux_face_values, aux_values)
    lift_scalings = rd.wf ./ rd.wq[vcat(rd.Fmask...)] # lift scalings for diag-norm SBP operators

    local_values_threaded = [Trixi.allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)
                             for _ in 1:Threads.nthreads()]

    # Use an array of SVectors (chunks of `nvars` are contiguous in memory) to speed up flux differencing
    fluxdiff_local_threaded = [zeros(SVector{nvars, uEltype}, rd.Nq)
                               for _ in 1:Threads.nthreads()]

    return (; md, Qrst_skew, dxidxhatj = md.rstxyzJ,
            invJ = inv.(md.J), lift_scalings, inv_wq = inv.(rd.wq),
            u_values, u_face_values, flux_face_values,
            aux_values, aux_quad_values, aux_face_values,
            local_values_threaded, fluxdiff_local_threaded)
end

# Version for sparse operators and symmetric fluxes with curved meshes
# @inline function Trixi.hadamard_sum!(du,
#                                A::LinearAlgebra.Adjoint{<:Any,
#                                                         <:AbstractSparseMatrixCSC},
#                                flux_is_symmetric::True, volume_flux,
#                                normal_directions::AbstractVector{<:AbstractVector},
#                                u, equations::AbstractCovariantEquations)
#     A_base = parent(A) # the adjoint of a SparseMatrixCSC is basically a SparseMatrixCSR
#     row_ids = axes(A, 2)
#     rows = rowvals(A_base)
#     vals = nonzeros(A_base)

#     for i in row_ids
#         u_i = u[i]
#         du_i = du[i]
#         for id in nzrange(A_base, i)
#             j = rows[id]
#             # This routine computes only the upper-triangular part of the hadamard sum (A .* F).
#             # We avoid computing the lower-triangular part, and instead accumulate those contributions
#             # while computing the upper-triangular part (using the fact that A is skew-symmetric and F
#             # is symmetric).
#             if j > i
#                 u_j = u[j]
#                 A_ij = vals[id]

#                 # provably entropy stable de-aliasing of geometric terms
#                 normal_direction = 0.5 * (getindex.(normal_directions, i) +
#                                     getindex.(normal_directions, j))

#                 AF_ij = 2 * A_ij * volume_flux(u_i, u_j, normal_direction, equations)
#                 du_i = du_i + AF_ij
#                 du[j] = du[j] - AF_ij
#             end
#         end
#         du[i] = du_i
#     end
# end