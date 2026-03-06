function DGMultiMeshCubedSphere2D(dg::DGMulti{NDIMS};
    initial_refinement = 3,
    is_on_boundary = nothing) where {NDIMS}

    NDIMS_AMBIENT = 3

    vertex_coordinates = calc_node_coordinates_cube_vertices(EARTH_RADIUS)

    Vxyz = ntuple(n -> vertex_coordinates[n, :], NDIMS_AMBIENT)

    EToV = zeros(Int, 6, 4)
    
    for i in 1:size(EToV, 1)
        EToV[i, :] = cubed_quadrilaterals_vertices_idx_map[i]
    end

    for j in 1:initial_refinement
        EToV_old = EToV
        Vxyz_old = ntuple(n -> copy(Vxyz[n]), NDIMS_AMBIENT)
        old_to_new = Dict{Int, Int}()
        edge_to_new = Dict{Tuple{Int, Int}, Int}()
        EToV = zeros(Int, (size(EToV_old, 1) * 4, 4))
        
        Vxyz = ntuple(n -> Vector{eltype(Vxyz[1])}(), NDIMS_AMBIENT)

        for i in 1:size(EToV_old, 1)
            idx_old = EToV_old[i, :]

            for k in idx_old
                if !haskey(old_to_new, k)
                    old_to_new[k] = length(Vxyz[1]) + 1
                    for n in 1:NDIMS_AMBIENT
                        push!(Vxyz[n], Vxyz_old[n][k])
                    end
                end
            end

            k_counter = 1
            for k in idx_old
                l_counter =  1
                for l in idx_old
                    if k < l && l_counter + k_counter != 5
                        edge = (k, l)
                        if !haskey(edge_to_new, edge)
                            edge_to_new[edge] = length(Vxyz[1]) + 1
                            vk = ntuple(n -> Vxyz_old[n][k], NDIMS_AMBIENT)
                            vl = ntuple(n -> Vxyz_old[n][l], NDIMS_AMBIENT)
                            midpoint = 0.5 .* (vk .+ vl)
                            midpoint = midpoint .* (EARTH_RADIUS / norm(midpoint)) # Normalize to outer radius
                            for n in 1:NDIMS_AMBIENT
                                push!(Vxyz[n], midpoint[n])
                            end
                        end
                    end
                    l_counter += 1
                end
                k_counter += 1
            end

            midpoint = ntuple(n -> 0.25 * sum(Vxyz_old[n][idx_old]), NDIMS_AMBIENT)
            midpoint = midpoint .* (EARTH_RADIUS / norm(midpoint)) # Normalize to outer radius
            for n in 1:NDIMS_AMBIENT
                push!(Vxyz[n], midpoint[n])
            end

            id1 = old_to_new[idx_old[1]]
            id2 = edge_to_new[Tuple(sort([idx_old[1], idx_old[2]]))]
            id3 = old_to_new[idx_old[2]]
            id4 = edge_to_new[Tuple(sort([idx_old[1], idx_old[3]]))]
            id5 = length(Vxyz[1])
            id6 = edge_to_new[Tuple(sort([idx_old[2], idx_old[4]]))]
            id7 = old_to_new[idx_old[3]]
            id8 = edge_to_new[Tuple(sort([idx_old[3], idx_old[4]]))]
            id9 = old_to_new[idx_old[4]]

            ids = [id1, id2, id3, id4, id5, id6, id7, id8, id9]

            # Fill EToV for the 4 new triangles
            for (sub_i, vertex_map) in enumerate(cubed_quad_vertices_idx_map)
                EToV[(i - 1) * 4 + sub_i, :] = getindex.(Ref(ids), vertex_map)
            end

        end
    end
    md = MeshData(Vxyz, EToV, dg.basis)
    # spherify_meshdata!(md, dg, EToV, Vxyz)
    norms = sqrt.(md.xyz[1].^2 .+ md.xyz[2].^2 .+ md.xyz[3].^2)
    md = @set md.xyz = map(c -> c ./ norms .* EARTH_RADIUS, md.xyz)
    norms = sqrt.(md.xyzq[1].^2 .+ md.xyzq[2].^2 .+ md.xyzq[3].^2)
    md = @set md.xyzq = map(c -> c ./ norms .* EARTH_RADIUS, md.xyzq)
    norms = sqrt.(md.xyzf[1].^2 .+ md.xyzf[2].^2 .+ md.xyzf[3].^2)
    md = @set md.xyzf = map(c -> c ./ norms .* EARTH_RADIUS, md.xyzf)
    boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)
    return DGMultiMesh(dg, Trixi.GeometricTermsType(Trixi.Curved(), dg), md, boundary_faces)
end

function spherify_meshdata!(md::MeshData, dg::DGMulti{NDIMS}, EToV, Vxyz) where {NDIMS}
    rd = dg.basis
    for e in 1:size(EToV, 1)
        v1, v2, v3, v4 = ntuple(n -> SVector(ntuple(d -> Vxyz[d][EToV[e, n]], 3)), 4)
        for j in 1:size(rd.rst[1], 1)
            r, s = rd.rst[1][j], rd.rst[2][j]
            # Bilinear mapping from reference square to physical space
            u_node = local_mapping(r, s, v1, v2, v4, v3, EARTH_RADIUS)

            for n in 1:3
                md = @set md.xyz[n][j, e] = u_node[n]
            end
        end

        for j in 1:size(rd.rstq[1], 1)
            r, s = rd.rstq[1][j], rd.rstq[2][j]
            # Bilinear mapping from reference square to physical space
            u_node = local_mapping(r, s, v1, v2, v4, v3, EARTH_RADIUS)

            for n in 1:3
                md = @set md.xyzq[n][j, e] = u_node[n]
            end
        end

        for j in 1:size(rd.rstf[1], 1)
            r, s = rd.rstf[1][j], rd.rstf[2][j]
            # Bilinear mapping from reference square to physical space
            u_node = local_mapping(r, s, v1, v2, v4, v3, EARTH_RADIUS)

            for n in 1:3
                md = @set md.xyzf[n][j, e] = u_node[n]
            end
        end
    end
    return md
end



# Function to compute the vertices' coordinates of an icosahedron inscribed in a sphere of radius `radius`
function calc_node_coordinates_cube_vertices(radius; RealT = Float64)
    vertices = Array{RealT, 2}(undef, 3, 12)

    vertices[:, 1] = [ 1/sqrt(3),  1/sqrt(3),  1/sqrt(3)]
    vertices[:, 2] = [-1/sqrt(3),  1/sqrt(3),  1/sqrt(3)]
    vertices[:, 3] = [ 1/sqrt(3), -1/sqrt(3),  1/sqrt(3)]
    vertices[:, 4] = [-1/sqrt(3), -1/sqrt(3),  1/sqrt(3)]
    vertices[:, 5] = [ 1/sqrt(3),  1/sqrt(3), -1/sqrt(3)]
    vertices[:, 6] = [-1/sqrt(3),  1/sqrt(3), -1/sqrt(3)]
    vertices[:, 7] = [ 1/sqrt(3), -1/sqrt(3), -1/sqrt(3)]
    vertices[:, 8] = [-1/sqrt(3), -1/sqrt(3), -1/sqrt(3)]

    return vertices * radius
end


# We use a local numbering to obtain the triangle vertices of each triangular face
#
# Fig: Local quad vertex numbering for a triangular face (corner vertices of the triangular face in parenthesis)
#                       5 (3)
#                      /\
#                     /  \
#                    /    \
#                   /      \
#                  /        \
#                 /          \
#                /            \
#               /              \
#              /                \
#            6/                 4\
#            /⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺⎺\
#           / \                 /  \
#          /   \               /    \
#         /     \             /      \
#        /       \           /        \
#       /         \         /          \ 
#      /           \       /            \
#     /             \     /              \
#    /               \   /                \
#  1/_________________\_/_________________3\
#   (1)                2                    (2)

# Index map for the vertices of each triangle on the triangular faces of the icosahedron (see Fig 4)
const cubed_quad_vertices_idx_map = ([1, 2, 4, 5], # Quad 1
                                     [2, 3, 5, 6], # Tri 2
                                     [4, 5, 7, 8], # Tri 3
                                     [5, 6, 8, 9]) # Tri 4

const cubed_quadrilaterals_vertices_idx_map = ([1, 2, 3, 4],
                                               [1, 5, 2, 6],
                                               [1, 3, 5, 7],
                                               [3, 4, 7, 8],
                                               [2, 4, 6, 8],
                                               [5, 6, 7, 8])