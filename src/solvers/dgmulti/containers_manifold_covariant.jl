@muladd begin
#! format: noindnent

# struct DGMultiAuxiliaryNodeVariableContainer{uEltype <: Real}
#     aux_node_vars::Array{uEltype, 3} # [variable, i, element]
#     aux_surface_node_vars::Array{uEltype, 4} # [variable, i, element]

#     # internal `resize!`able storage
#     _aux_node_vars::Vector{uEltype}
#     _aux_surface_node_vars::Vector{uEltype}
# end

# function init_auxiliary_node_variables(mesh::DGMultiMesh, equations::AbstractCovariantEquations,
#                                   dg::DGMulti, auxiliary_field)
#     @unpack md = mesh
#     nelements = Trixi.ncells(mesh)
#     ninterfaces = size(md.FToF, 1) # may include boundary interfaces
#     NDIMS = 3 # TODO: get ndims properly
#     uEltype = Float64

#     rd = dg.basis

#     # wedges can have different numbers of face nodes, so we need find the maximum and leave 
#     # leftover elements somehow empty
#     Nf_max = maximum(length, rd.element_type.node_ids_by_face)

#     _aux_node_vars = Vector{uEltype}(undef,
#                                      n_aux_node_vars(equations) *
#                                      rd.Nq * nelements)
#     aux_node_vars = Trixi.unsafe_wrap(Array, pointer(_aux_node_vars),
#                                       (n_aux_node_vars(equations),
#                                        rd.Nq,
#                                        nelements))
#     println("Length check aux_node_vars: ",
#         length(_aux_node_vars), " == ", prod(size(aux_node_vars)))
#     _aux_surface_node_vars = Vector{uEltype}(undef,
#                                              2 * n_aux_node_vars(equations) *
#                                              Nf_max *
#                                              ninterfaces)
#     aux_surface_node_vars = Trixi.unsafe_wrap(Array,
#                                               pointer(_aux_surface_node_vars),
#                                               (2, n_aux_node_vars(equations),
#                                               Nf_max, 
#                                               ninterfaces))

#     println("Length check aux_surface_node_vars: ",
#         length(_aux_surface_node_vars), " == ", prod(size(aux_surface_node_vars)))

#     auxiliary_variables = DGMultiAuxiliaryNodeVariableContainer{uEltype}(aux_node_vars,
#                                                                          aux_surface_node_vars,
#                                                                          _aux_node_vars,
#                                                                          _aux_surface_node_vars)

#     init_auxiliary_node_variables!(auxiliary_variables, mesh, equations, dg,
#                                    md, auxiliary_field)
    
#     #init_auxiliary_surface_node_variables!(auxiliary_variables, mesh, equations, dg)
#     return auxiliary_variables
# end

function init_auxiliary_node_variables!(aux_values, aux_quad_values, mesh::DGMultiMesh,
                                        equations::AbstractCovariantEquations{2, 3}, dg,
                                        bottom_topography)
    rd = dg.basis
    (; line, tri) = rd.approximation_type
    (; xyz) = mesh.md
    md = mesh.md
    n_aux = n_aux_node_vars(equations)
    # Check that the degree of the mesh matches that of the solver
    # @assert length(mesh.nodes) == nnodes(dg)

    # The tree node coordinates are assumed to be on the spherical shell centred around the 
    # origin. Therefore, we can compute the radius once and use it throughout.
    # radius = norm(Trixi.get_node_coords(tree_node_coordinates, equations, dg, 1, 1, 1))

    for element in eachelement(mesh, dg)
        if size(xyz[1][:, element])[1] != 6

               error("The mesh element must have 6 corner vertices, but found $(size(xyz[1][:, element]))")
        end
        # Extract the corner vertex positions from the Mesh
        # TODO: Find a way to extract corner vertices when plotting nodes are not excusively the corner vertices
        v1 = [xyz[1][4, element], xyz[2][4, element], xyz[3][4, element]]
        v2 = [xyz[1][5, element], xyz[2][5, element], xyz[3][5, element]]
        v3 = [xyz[1][6, element], xyz[2][6, element], xyz[3][6, element]]
        v1, v2, v3 = v1, v3, v2
        radius = norm(v3)

        aux_node = Vector{eltype(aux_values[1, 1])}(undef, n_aux)
        
        # Compute the auxiliary metric information at each node
        for i in 1:Trixi.nnodes(dg)
            # Covariant basis in the desired global coordinate system as columns of a matrix
            basis_covariant = calc_basis_covariant(v1, v2, v3,
                                                   rd.rst[2][i], rd.rst[3][i],
                                                   radius,
                                                   equations.global_coordinate_system,
                                                   equations)
            
            aux_node[1:6] = SVector(basis_covariant)
            
            
            # Covariant metric tensor G := basis_covariant' * basis_covariant
            metric_covariant = basis_covariant' * basis_covariant

            # Contravariant metric tensor inv(G)
            metric_contravariant = inv(metric_covariant)

            # Contravariant basis vectors as rows of a matrix
            basis_contravariant = metric_contravariant * basis_covariant'


            aux_node[7:12] = SVector(basis_contravariant)
            # Area element
            aux_node[13] = sqrt(det(metric_covariant))

            # Covariant metric tensor components
            aux_node[14:16] = SVector(metric_covariant[1, 1],
                                                          metric_covariant[1, 2],
                                                          metric_covariant[2, 2])

            # Contravariant metric tensor components
            aux_node[17:19] = SVector(metric_contravariant[1, 1],
                                                          metric_contravariant[1, 2],
                                                          metric_contravariant[2, 2])
            # Bottom topography
            if !isnothing(bottom_topography)
                nothing
            end
            aux_values[i, element] = SVector{n_aux}(aux_node)
        end
        # Christoffel symbols of the second kind (aux_values[21:26, :, :, element])
        calc_christoffel_symbols!(aux_values, mesh, equations, dg, element)

        # Compute the auxiliary metric information at each node
        for i in Trixi.each_quad_node(mesh, dg)
            # Covariant basis in the desired global coordinate system as columns of a matrix
            basis_covariant = calc_basis_covariant(v1, v2, v3,
                                                   rd.rstq[2][i], rd.rstq[3][i],
                                                   radius,
                                                   equations.global_coordinate_system, equations)
            
            aux_node[1:6] = SVector(basis_covariant)
            
            
            # Covariant metric tensor G := basis_covariant' * basis_covariant
            metric_covariant = basis_covariant' * basis_covariant

            # Contravariant metric tensor inv(G)
            metric_contravariant = inv(metric_covariant)

            # Contravariant basis vectors as rows of a matrix
            basis_contravariant = metric_contravariant * basis_covariant'
            aux_node[7:12] = SVector(basis_contravariant)
            
            # Area element
            aux_node[13] = sqrt(det(metric_covariant))
            
            # Covariant metric tensor components
            aux_node[14:16] = SVector(metric_covariant[1, 1],
                                                          metric_covariant[1, 2],
                                                          metric_covariant[2, 2])

            # Contravariant metric tensor components
            aux_node[17:19] = SVector(metric_contravariant[1, 1],
                                                          metric_contravariant[1, 2],
                                                          metric_contravariant[2, 2])
            # Bottom topography
            if !isnothing(bottom_topography)
                nothing
            end
            aux_quad_values[i, element] = SVector{n_aux}(aux_node)
        end
    end

    return nothing
end

# Analytically compute the transformation matrix A, such that G = AᵀA is the 
# covariant metric tensor and a_i = A[1,i] * e_x + A[2,i] * e_y + A[3,i] * e_z denotes 
# the covariant tangent basis, where e_x, e_y, and e_z are the Cartesian unit basis vectors.
@inline function calc_basis_covariant(v1, v2, v3, xi2, xi3, radius,
                                      ::GlobalCartesianCoordinates, equations)
    # Construct a bilinear mapping based on the four corner vertices
    xe = 0.5f0 * (-(xi2 + xi3) * v1 + (1 + xi2) * v2 +
          (1 + xi3) * v3)
    # Derivatives of bilinear map with respect to reference coordinates xi1, xi2
    dxedxi2 = 0.5f0 *
              (-v1 + v2)
    dxedxi3 = 0.5f0 *
              (-v1 + v3)

    if equations.flat
        return SMatrix{3, 2}(dxedxi2[1], dxedxi2[2], dxedxi2[3],
                             dxedxi3[1], dxedxi3[2], dxedxi3[3])
    end

    # Use product/quotient rule on the projection
    norm_xe = norm(xe)
    dxdxi2 = radius / norm_xe * (dxedxi2 - dot(xe, dxedxi2) / norm_xe^2 * xe)
    dxdxi3 = radius / norm_xe * (dxedxi3 - dot(xe, dxedxi3) / norm_xe^2 * xe)

    return SMatrix{3, 2}(dxdxi2[1], dxdxi2[2], dxdxi2[3],
                         dxdxi3[1], dxdxi3[2], dxdxi3[3])
end

# Analytically compute the transformation matrix A, such that G = AᵀA is the 
# covariant metric tensor and a_i = A[1,i] * e_lon + A[2,i] * e_lat denotes 
# the covariant tangent basis, where e_lon and e_lat are the unit basis vectors
# in the longitudinal and latitudinal directions, respectively. This formula is 
# taken from Guba et al. (2014).
@inline function calc_basis_covariant(v1, v2, v3, xi2, xi3, radius,
                                      ::GlobalSphericalCoordinates)
    # Construct a bilinear mapping based on the four corner vertices
    xe = 0.5f0 * ((2 - (1 + xi2) - (1 + xi3)) * v1 + (1 + xi2) * v2 +
          (1 + xi3) * v3)
    # Project the mapped local coordinates onto the sphere using a simple scaling
    scaling_factor = radius / norm(xe)
    x = scaling_factor * xe

    # Convert Cartesian coordinates to longitude and latitude
    lon, lat = atan(x[2], x[1]), asin(min(x[3] / radius, 1))

    # Compute trigonometric terms needed for analytical metrics
    sinlon, coslon = sincos(lon)
    sinlat, coslat = sincos(lat)
    a11 = sinlon * sinlon * coslat * coslat + sinlat * sinlat
    a12 = a21 = -sinlon * coslon * coslat * coslat
    a13 = -coslon * sinlat * coslat
    a22 = coslon * coslon * coslat * coslat + sinlat * sinlat
    a23 = -sinlon * sinlat * coslat
    a31 = -coslon * sinlat
    a32 = -sinlon * sinlat
    a33 = coslat

    # Compute the matrix A containing spherical components of the covariant basis
    A = 0.5f0 * scaling_factor *
        SMatrix{2, 3}(-sinlon, 0, coslon, 0, 0, 1) *
        SMatrix{3, 3}(a11, a21, a31, a12, a22, a32, a13, a23, a33) *
        SMatrix{3, 3}(v1[1], v1[2], v1[3], v2[1], v2[2], v2[3],
                      v3[1], v3[2], v3[3]) *
        SMatrix{3, 2}(-1 , 1, 0,
                      -1 , 0, 1)

    # Make zero component in the radial direction so the matrix has the right dimensions
    return SMatrix{3, 2}(A[1, 1], A[2, 1], 0.0f0, A[1, 2], A[2, 2], 0.0f0)
end

function calc_christoffel_symbols!(aux_values, mesh::DGMultiMesh,
                                   equations::AbstractCovariantEquations{2, 3}, dg,
                                   element)
    rd = dg.basis
    (; Vq, Drst) = rd

    # Compute differentiation matrices for ξ¹ (s) and ξ² (t)
    # TODO: This is probably wrong
    # Dsq, Dtq = map(D -> Vq * D, Drst[2:3])
    Dsq, Dtq = map(D -> D, Drst[2:3])

    for i in 1:Trixi.nnodes(dg)

        # Numerically differentiate covariant metric components with respect to ξ¹
        dG11dxi1 = zero(eltype(aux_values[i, element]))
        dG12dxi1 = zero(eltype(aux_values[i, element]))
        dG22dxi1 = zero(eltype(aux_values[i, element]))
        for jj in 1:nnodes(dg)
            aux_node_jj = aux_values[jj, element]
            Gcov_jj = metric_covariant(aux_node_jj, equations)
            dG11dxi1 += Dsq[i, jj] * Gcov_jj[1, 1]
            dG12dxi1 += Dsq[i, jj] * Gcov_jj[1, 2]
            dG22dxi1 += Dsq[i, jj] * Gcov_jj[2, 2]
        end

        # Numerically differentiate covariant metric components with respect to ξ²
        dG11dxi2 = zero(eltype(aux_values[i, element]))
        dG12dxi2 = zero(eltype(aux_values[i, element]))
        dG22dxi2 = zero(eltype(aux_values[i, element]))
        for jj in 1:nnodes(dg)
            aux_node_jj = aux_values[jj, element]
            Gcov_jj = metric_covariant(aux_node_jj, equations)
            dG11dxi2 += Dtq[i, jj] * Gcov_jj[1, 1]
            dG12dxi2 += Dtq[i, jj] * Gcov_jj[1, 2]
            dG22dxi2 += Dtq[i, jj] * Gcov_jj[2, 2]
        end

        # Compute Christoffel symbols of the first kind
        christoffel_firstkind_1 = SMatrix{2, 2}(0.5f0 * dG11dxi1,
                                                0.5f0 * dG11dxi2,
                                                0.5f0 * dG11dxi2,
                                                dG12dxi2 - 0.5f0 * dG22dxi1)
        christoffel_firstkind_2 = SMatrix{2, 2}(dG12dxi1 - 0.5f0 * dG11dxi2,
                                                0.5f0 * dG22dxi1,
                                                0.5f0 * dG22dxi1,
                                                0.5f0 * dG22dxi2)

        # Raise indices to get Christoffel symbols of the second kind
        aux_node = Vector(aux_values[i, element])
        Gcon = metric_contravariant(aux_node, equations)
        aux_node[21] = Gcon[1, 1] * christoffel_firstkind_1[1, 1] +
                                           Gcon[1, 2] * christoffel_firstkind_2[1, 1]
        aux_node[22] = Gcon[1, 1] * christoffel_firstkind_1[1, 2] +
                                           Gcon[1, 2] * christoffel_firstkind_2[1, 2]
        aux_node[23] = Gcon[1, 1] * christoffel_firstkind_1[2, 2] +
                                           Gcon[1, 2] * christoffel_firstkind_2[2, 2]

        aux_node[24] = Gcon[2, 1] * christoffel_firstkind_1[1, 1] +
                                           Gcon[2, 2] * christoffel_firstkind_2[1, 1]
        aux_node[25] = Gcon[2, 1] * christoffel_firstkind_1[1, 2] +
                                           Gcon[2, 2] * christoffel_firstkind_2[1, 2]
        aux_node[26] = Gcon[2, 1] * christoffel_firstkind_1[2, 2] +
                                           Gcon[2, 2] * christoffel_firstkind_2[2, 2]
        aux_values[i, element] = SVector{n_aux_node_vars(equations)}(aux_node)
    end
end
end # @muladd