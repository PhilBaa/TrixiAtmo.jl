# Constructs cache variables for both affine and non-affine (curved) DGMultiMeshes
function Trixi.create_cache(mesh::DGMultiMesh{NDIMS}, equations::AbstractCovariantEquations, dg::Trixi.DGMultiWeakForm, 
                            metric_terms, auxiliary_field, RealT,
                      uEltype) where {NDIMS}
    rd = dg.basis
    md = mesh.md


    x1 = ntuple(n -> md.xyz[n][1, 1], 3) 
    x4 = ntuple(n -> md.xyz[n][4, 1], 3)
    outer_radius = norm(x4) - norm(x1)

    # volume quadrature weights, volume interpolation matrix, mass matrix, differentiation matrices
    @unpack wq, Vq, M, Drst = rd

    # ∫f(u) * dv/dx_i = ∑_j (Vq*Drst[i])'*diagm(wq)*(rstxyzJ[i,j].*f(Vq*u))
    weak_differentiation_matrices = map(D -> -M \ ((Vq * D)' * Diagonal(wq)), Drst)

    nvars = nvariables(equations)
    naux = n_aux_node_vars(equations)

    # storage for volume quadrature values, face quadrature values, flux values
    u_values = Trixi.allocate_nested_array(uEltype, nvars, size(md.xq), dg)
    u_face_values = Trixi.allocate_nested_array(uEltype, nvars, size(md.xf), dg)
    flux_face_values = Trixi.allocate_nested_array(uEltype, nvars, size(md.xf), dg)
    aux_values = Trixi.allocate_nested_array(uEltype, naux, size(md.x), dg)
    aux_quad_values = Trixi.allocate_nested_array(uEltype, naux, size(md.xq), dg)
    aux_face_values = Trixi.allocate_nested_array(uEltype, naux, size(md.xf), dg)
    
    if typeof(rd.approximation_type) <:
       Union{SBP, Trixi.AbstractNonperiodicDerivativeOperator}
        lift_scalings = rd.wf ./ rd.wq[rd.Fmask] # lift scalings for diag-norm SBP operators
    else
        lift_scalings = nothing
    end

    # local storage for volume integral and source computations
    local_values_threaded = [Trixi.allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)
                             for _ in 1:Threads.nthreads()]

    # For curved meshes, we interpolate geometric terms from nodal points to quadrature points.
    # For affine meshes, we just access one element of this interpolated data.
    dxidxhatj = map(x -> rd.Vq * x, md.rstxyzJ)

    init_auxiliary_node_variables!(aux_values, aux_quad_values, mesh, equations, dg, auxiliary_field)

    # Interpolate auxiliary variables to quadrature and face points
    Trixi.apply_to_each_field(Trixi.mul_by!(rd.Vf), aux_face_values, aux_values)


    # interpolate J to quadrature points for weight-adjusted DG (WADG)
    invJ = inv.(rd.Vq * md.J)

    # for scaling by curved geometric terms (not used by affine DGMultiMesh)
    flux_threaded = [[Trixi.allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)
                      for _ in 1:NDIMS] for _ in 1:Threads.nthreads()]
    rotated_flux_threaded = [Trixi.allocate_nested_array(uEltype, nvars, (rd.Nq,), dg)
                             for _ in 1:Threads.nthreads()]

    cache = (; md, weak_differentiation_matrices, lift_scalings, invJ, dxidxhatj,
            u_values, u_face_values, flux_face_values,
            aux_values, aux_quad_values, aux_face_values,
            local_values_threaded, flux_threaded, rotated_flux_threaded,
            outer_radius) # TODO: outer radius should probably be handled differently
    return cache
end

function create_cache_auxiliary(mesh::DGMultiMesh, equations, have_aux_node_vars::True, dg, auxiliary_field)
    # initialize auxiliary variables
    auxiliary_variables = init_auxiliary_node_variables(mesh, equations, dg, auxiliary_field)
    return (; auxiliary_variables)
end

function create_cache_auxiliary(mesh::DGMultiMesh, equations, have_aux_node_vars::False, dg, auxiliary_field)
    return NamedTuple()
end

include("containers_manifold_covariant.jl")
include("dg_manifold_covariant.jl")
