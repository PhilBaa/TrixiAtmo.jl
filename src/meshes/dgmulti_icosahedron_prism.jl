function DGMultiMeshPrismIcosahedron3D(dg::DGMulti{3, <:Wedge}, inner_radius, outer_radius;
                                       horizontal_initial_refinement = 3,
                                       vertical_layers = 5,
                                       is_on_boundary = nothing)
    NDIMS_AMBIENT = 3

    radius = 1.0

    vertex_coordinates = calc_node_coordinates_icosahedron_vertices(radius)

    Vxyz_tri = ntuple(n -> vertex_coordinates[n, :], NDIMS_AMBIENT)

    # We first create a 2D mesh of the icosahedron surface, and then extrude it to create the prism elements.
    # The EToV connectivity will be the same for the triangles on the surface.
    EToV_tri = zeros(Int, 20, 3)

    for i in 1:size(EToV_tri, 1)
        EToV_tri[i, :] = icosahedron_triangle_vertices_idx_map[i]
    end

    for j in 1:horizontal_initial_refinement
        EToV_old = EToV_tri
        Vxyz_old = ntuple(n -> copy(Vxyz_tri[n]), NDIMS_AMBIENT)
        old_to_new = Dict{Int, Int}()
        edge_to_new = Dict{Tuple{Int, Int}, Int}()
        EToV_tri = zeros(Int, (size(EToV_old, 1) * 4, 3))

        Vxyz_tri = ntuple(n -> Vector{eltype(Vxyz_tri[1])}(), NDIMS_AMBIENT)

        for i in 1:size(EToV_old, 1)
            idx_old = EToV_old[i, :]

            for k in idx_old
                if !haskey(old_to_new, k)
                    old_to_new[k] = length(Vxyz_tri[1]) + 1
                    for n in 1:NDIMS_AMBIENT
                        push!(Vxyz_tri[n], Vxyz_old[n][k])
                    end
                end
            end

            for k in idx_old, l in idx_old
                if k < l
                    edge = (k, l)
                    if !haskey(edge_to_new, edge)
                        edge_to_new[edge] = length(Vxyz_tri[1]) + 1
                        vk = ntuple(n -> Vxyz_old[n][k], NDIMS_AMBIENT)
                        vl = ntuple(n -> Vxyz_old[n][l], NDIMS_AMBIENT)
                        midpoint = 0.5 .* (vk .+ vl)
                        midpoint = midpoint .* (radius / norm(midpoint)) # Normalize to outer radius
                        for n in 1:NDIMS_AMBIENT
                            push!(Vxyz_tri[n], midpoint[n])
                        end
                    end
                end
            end
            id1 = old_to_new[idx_old[1]]
            id2 = edge_to_new[Tuple(sort([idx_old[1], idx_old[2]]))]
            id3 = old_to_new[idx_old[2]]
            id4 = edge_to_new[Tuple(sort([idx_old[2], idx_old[3]]))]
            id5 = old_to_new[idx_old[3]]
            id6 = edge_to_new[Tuple(sort([idx_old[3], idx_old[1]]))]

            ids = [id1, id2, id3, id4, id5, id6]

            # Fill EToV for the 4 new triangles
            for (sub_i, vertex_map) in enumerate(icosahedron_tri_vertices_idx_map)
                EToV_tri[(i - 1) * 4 + sub_i, :] = getindex.(Ref(ids), vertex_map)
            end
        end
    end

    # Create the prism elements by extruding the 2D mesh
    num_triangles = size(EToV_tri, 1)
    num_points_per_layer = size(Vxyz_tri[1], 1)
    EToV = zeros(Int, num_triangles * vertical_layers, 6)
    Vxyz = ntuple(n -> Vxyz_tri[n] .* inner_radius, NDIMS_AMBIENT)
    radius_list = range(inner_radius, outer_radius, length=vertical_layers + 1)

    for i in 1:vertical_layers
        EToV[(i - 1) * num_triangles + 1:i * num_triangles, 1:3] = EToV_tri .+ (i - 1) * num_points_per_layer
        EToV[(i - 1) * num_triangles + 1:i * num_triangles, 4:6] = EToV_tri .+ i * num_points_per_layer
        Vxyz_layer = ntuple(n -> Vxyz_tri[n] .* radius_list[i + 1], NDIMS_AMBIENT)
        Vxyz = ntuple(n -> vcat(Vxyz[n], Vxyz_layer[n]), NDIMS_AMBIENT)
    end
    md = StartUpDG.MeshData(Vxyz, EToV, dg.basis)
    project_onto_sphere!(md, dg)
    boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)
    return DGMultiMesh(dg, Trixi.GeometricTermsType(Trixi.Curved(), dg), md, boundary_faces)
end

function project_onto_sphere!(md::MeshData, dg::DGMulti{NDIMS, <:Wedge}) where {NDIMS}
    rd = dg.basis
    (; xyz, xyzq, xyzf) = md
    for e in 1:size(md.xyz[1], 2)
        VX, VY, VZ = map(coords -> transpose(coords[:, e]) / rd.V1', md.xyz)
        vertices = ntuple(n -> [VX[n], VY[n], VZ[n]], 6)
        inner_radius = norm(vertices[1])
        outer_radius = norm(vertices[4])
        for j in 1:size(rd.rst[1], 1)
            t = rd.rst[3][j]
            radius = (1 - t) / 2 * inner_radius + (1 + t) / 2 * outer_radius
            x_node = ntuple(n -> xyz[n][j, e], NDIMS)
            x_node = radius / norm(x_node) .* x_node

            for n in 1:3
                xyz[n][j, e] = x_node[n]
            end
        end

        for j in 1:size(rd.rstq[1], 1)
            r, s, t = rd.rstq[1][j], rd.rstq[2][j], rd.rstq[3][j]
            radius = (1 - t) / 2 * inner_radius + (1 + t) / 2 * outer_radius
            x_node = ntuple(n -> xyzq[n][j, e], NDIMS)
            x_node = radius / norm(x_node) .* x_node

            for n in 1:3
                xyzq[n][j, e] = x_node[n]
            end
        end

        for j in 1:size(rd.rstf[1], 1)
            r, s, t = rd.rstf[1][j], rd.rstf[2][j], rd.rstf[3][j]
            radius = (1 - t) / 2 * inner_radius + (1 + t) / 2 * outer_radius
            x_node = ntuple(n -> xyzf[n][j, e], NDIMS)
            x_node = radius / norm(x_node) .* x_node

            for n in 1:3
                xyzf[n][j, e] = x_node[n]
            end
        end
    end
    md = setproperties(md, xyz = xyz, xyzq = xyzq, xyzf = xyzf)
    return md
end
