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

function Trixi.compute_coefficients!(::Nothing, u, initial_condition, t,
                                     mesh::DGMultiMesh, equations::AbstractCovariantEquations,
                                     dg::Trixi.DGMultiFluxDiffSBP, cache)
    md = mesh.md
    rd = dg.basis
    @unpack u_values, aux_quad_values = cache
    # evaluate the initial condition at quadrature points
    Trixi.@threaded for i in Trixi.each_quad_node_global(mesh, dg, cache)
        x_node = SVector(getindex.(md.xyzq, i))
        aux_node = aux_quad_values[i]
        u_values[i] = initial_condition(x_node, t, aux_node, equations)
    end

    # multiplying by Pq computes the L2 projection
    u .= u_values
end
