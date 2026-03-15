function Trixi.create_cache(mesh::DGMultiMesh, equations::AbstractCovariantEquations, dg::Trixi.DGMultiFluxDiffSBP,
                            RealT, metric_terms, auxiliary_field,
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

    invJ = inv.(area_element.(aux_quad_values, equations))

    return (; md, Qrst_skew, dxidxhatj = md.rstxyzJ,
            invJ = invJ, lift_scalings, inv_wq = inv.(rd.wq),
            u_values, u_face_values, flux_face_values,
            aux_values, aux_quad_values, aux_face_values,
            local_values_threaded, fluxdiff_local_threaded)
end

function Trixi.calc_volume_integral!(du, u, mesh::DGMultiMesh, have_nonconservative_terms,
                                     equations::AbstractCovariantEquations,
                                     volume_integral, dg::Trixi.DGMultiFluxDiffSBP,
                                     cache)
    @unpack fluxdiff_local_threaded, inv_wq, aux_values = cache

    Trixi.@threaded for e in eachelement(mesh, dg, cache)
        fluxdiff_local = fluxdiff_local_threaded[Threads.threadid()]
        fill!(fluxdiff_local, zero(eltype(fluxdiff_local)))
        u_local = view(u, :, e)
        aux_local = view(aux_values, :, e)

        Trixi.local_flux_differencing!(fluxdiff_local, u_local, aux_local, e,
                                       have_nonconservative_terms,
                                       volume_integral.volume_flux,
                                       Trixi.has_sparse_operators(dg),
                                       mesh, equations, dg, cache)

        for i in Trixi.each_quad_node(mesh, dg, cache)
            du[i, e] = du[i, e] + fluxdiff_local[i] * inv_wq[i]
        end
    end
end

# Specialize since `u_values` isn't computed for DGMultiFluxDiffSBP solvers.
function Trixi.calc_sources!(du, u, t, source_terms,
                             mesh, equations::AbstractCovariantEquations,
                             dg::Trixi.DGMultiFluxDiffSBP, cache)
    md = mesh.md
    @unpack aux_values = cache

    Trixi.@threaded for e in Trixi.eachelement(mesh, dg, cache)
        for i in Trixi.each_quad_node(mesh, dg, cache)
            du[i, e] += source_terms(u[i, e], SVector(getindex.(md.xyzq, i, e)), t,
                                     aux_values[i, e], equations)
        end
    end
end

function Trixi.calc_sources!(du, u, t, source_term::Nothing,
                             mesh, equations::AbstractCovariantEquations,
                             dg::Trixi.DGMultiFluxDiffSBP, cache)
    nothing
end

# Computes flux differencing contribution from each Cartesian direction over a single element.
# For dense operators, we do not use sum factorization.
@inline function Trixi.local_flux_differencing!(fluxdiff_local, u_local, aux_local, element_index,
                                                has_nonconservative_terms::False, volume_flux,
                                                has_sparse_operators::False, mesh,
                                                equations::AbstractCovariantEquations,
                                                dg::DGMulti{NDIMS}, cache) where {NDIMS}
    for dim in 1:NDIMS
        Qi_skew = cache.Qrst_skew[dim]
        # True() indicates the volume flux is symmetric
        Trixi.hadamard_sum!(fluxdiff_local, Qi_skew,
                            True(), volume_flux,
                            dim, u_local, aux_local, equations)
    end
end

# When the operators are sparse, we use the sum-factorization approach to
# computing flux differencing.
@inline function Trixi.local_flux_differencing!(fluxdiff_local, u_local, aux_local, element_index,
                                                has_nonconservative_terms::False, volume_flux,
                                                has_sparse_operators::True, mesh,
                                                equations, dg::DGMulti{NDIMS}, cache) where {NDIMS}
    @unpack Qrst_skew = cache
    for dim in 1:NDIMS
        Qi_skew = Qrst_skew[dim]

        # True() indicates the flux is symmetric
        Trixi.hadamard_sum!(fluxdiff_local, Qi_skew,
                            True(), volume_flux,
                            dim, u_local, aux_local, equations)
    end
end

@inline function Trixi.local_flux_differencing!(fluxdiff_local, u_local, aux_local, element_index,
                                                has_nonconservative_terms::True, volume_flux,
                                                has_sparse_operators::False, mesh,
                                                equations, dg::DGMulti{NDIMS}, cache) where {NDIMS}
    flux_conservative, flux_nonconservative = volume_flux
    for dim in 1:NDIMS
        Qi_skew = cache.Qrst_skew[dim]
        # True() indicates the flux is symmetric.
        Trixi.hadamard_sum!(fluxdiff_local, Qi_skew,
                      True(), flux_conservative,
                      dim, u_local, aux_local, equations)

        # The final argument .5 scales the operator by 1/2 for the nonconservative terms.
        half_Qi_skew = Trixi.LazyMatrixLinearCombo((cache.Qrst_skew[dim], ), (0.5, ))
        # False() indicates the flux is non-symmetric.
        Trixi.hadamard_sum!(fluxdiff_local, half_Qi_skew,
                      False(), flux_nonconservative,
                      dim, u_local, aux_local, equations)
    end
end

# Version for dense operators and symmetric fluxes
@inline function Trixi.hadamard_sum!(du, A,
                                     flux_is_symmetric::True, volume_flux,
                                     orientation_or_normal_direction,
                                     u, aux, equations::AbstractCovariantEquations)
    row_ids, col_ids = axes(A)

    for i in row_ids
        u_i = u[i]
        aux_i = aux[i]
        du_i = du[i]
        for j in col_ids
            # This routine computes only the upper-triangular part of the hadamard sum (A .* F).
            # We avoid computing the lower-triangular part, and instead accumulate those contributions
            # while computing the upper-triangular part (using the fact that A is skew-symmetric and F
            # is symmetric).
            if j > i
                u_j = u[j]
                aux_j = aux[j]
                AF_ij = 2 * A[i, j] *
                        volume_flux(u_i, u_j, aux_i, aux_j, orientation_or_normal_direction,
                                    equations)
                du_i = du_i + AF_ij
                du[j] = du[j] - AF_ij
            end
        end
        du[i] = du_i
    end
end

# Version for dense operators and non-symmetric fluxes
@inline function Trixi.hadamard_sum!(du, A,
                                     flux_is_symmetric::False, volume_flux,
                                     orientation::Integer,
                                     u, aux, equations::AbstractCovariantEquations)
    row_ids, col_ids = axes(A)

    for i in row_ids
        u_i = u[i]
        aux_i = aux[i]
        du_i = du[i]
        for j in col_ids
            u_j = u[j]
            aux_j = aux[j]
            f_ij = volume_flux(u_i, u_j, aux_i, aux_j, orientation, equations)
            du_i = du_i + 2 * A[i, j] * f_ij
        end
        du[i] = du_i
    end
end
