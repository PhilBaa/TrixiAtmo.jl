function Trixi.compute_coefficients!(u, initial_condition, t,
                               mesh::DGMultiMesh, equations::AbstractCovariantEquations, dg::DGMulti, cache)
    md = mesh.md
    rd = dg.basis
    @unpack u_values, aux_quad_values = cache
    # evaluate the initial condition at quadrature points
    Trixi.@threaded for e in eachelement(mesh, dg, cache)
        for i in Trixi.each_quad_node(mesh, dg)
            x_node = SVector(ntuple(k-> md.xyzq[k][i, e], 3))
            
            aux_node = aux_quad_values[i, e]

            u_values[i, e] = initial_condition(x_node, t, aux_node, equations)        
        end
    end

    # multiplying by Pq computes the L2 projection
    Trixi.apply_to_each_field(Trixi.mul_by!(rd.Pq), u, u_values)
end

# version for curved meshes
function Trixi.calc_volume_integral!(du, u, mesh::DGMultiMesh,
                               have_nonconservative_terms::False,
                               equations::AbstractCovariantEquations{NDIMS},
                               volume_integral::VolumeIntegralWeakForm, dg::DGMulti,
                               cache) where {NDIMS_AMBIENT, NDIMS}
    rd = dg.basis
    (; weak_differentiation_matrices, u_values) = cache
    (; aux_quad_values) = cache
    (; dxidxhatj) = cache

    # interpolate to quadrature points
    Trixi.apply_to_each_field(Trixi.mul_by!(rd.Vq), u_values, u)

    Trixi.@threaded for e in eachelement(mesh, dg, cache)
        flux_values = cache.flux_threaded[Threads.threadid()]
        for i in 1:NDIMS    # specialization for 2D covariant equations
            # Here, the broadcasting operation does not allocate
            # TODO: Now this is completely wrong and just needs to be done differently
            # old code
            # flux_values[i] .= flux.(view(u_values, :, e), i, equations)
            for j in Trixi.each_quad_node(mesh, dg)
                u_node = u_values[j, e]
                aux_node = aux_quad_values[j, e]
                flux_values[i][j] =
                    flux(u_node, aux_node, i, equations)
            end           
        end

        # rotate flux with df_i/dx_i = sum_j d(x_i)/d(x̂_j) * d(f_i)/d(x̂_j).
        # Example: df_x/dx + df_y/dy = dr/dx * df_x/dr + ds/dx * df_x/ds
        #                  + dr/dy * df_y/dr + ds/dy * df_y/ds
        #                  = Dr * (dr/dx * fx + dr/dy * fy) + Ds * (...)
        #                  = Dr * (f_r) + Ds * (f_s)

        rotated_flux_values = cache.rotated_flux_threaded[Threads.threadid()]
        for j in Trixi.eachdim(mesh)
            fill!(rotated_flux_values, zero(eltype(rotated_flux_values)))

            # compute rotated fluxes
            for i in Trixi.eachdim(mesh)
                for ii in eachindex(rotated_flux_values)
                    flux_i_node = flux_values[i][ii]
                    dxidxhatj_node = dxidxhatj[i, j][ii, e]
                    rotated_flux_values[ii] = rotated_flux_values[ii] +
                                              dxidxhatj_node * flux_i_node
                end
            end

            # apply weak differentiation matrices to rotated fluxes
            Trixi.apply_to_each_field(Trixi.mul_by_accum!(weak_differentiation_matrices[j]),
                                view(du, :, e), rotated_flux_values)
        end
    end
end

function Trixi.calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm,
                              mesh::DGMultiMesh,
                              have_nonconservative_terms::False,
                              equations::AbstractCovariantEquations{NDIMS},
                              dg::DGMulti{NDIMS_AMBIENT}) where {NDIMS_AMBIENT, NDIMS}
    @unpack surface_flux = surface_integral
    md = mesh.md
    @unpack mapM, mapP, nxyzJ, xyzf, Jf = md
    @unpack u_face_values, flux_face_values, aux_face_values, outer_radius = cache

    Trixi.@threaded for face_node_index in Trixi.each_face_node_global(mesh, dg, cache)

        # inner (idM -> minus) and outer (idP -> plus) indices
        idM, idP = mapM[face_node_index], mapP[face_node_index]
        uM = u_face_values[idM]
        uP = u_face_values[idP]
        auxM = aux_face_values[idM]
        auxP = aux_face_values[idP]
        # Transform uP to the same coordinate system as uM
        uP_global = contravariant2global(uP, auxP, equations)
        uP_transformed_to_M = global2contravariant(uP_global, auxM, equations)

        # compute the normal vector at the face
        normal = SVector{NDIMS_AMBIENT}(getindex.(nxyzJ, idM)) / Jf[idM] 
        # TODO: make this more general, right now we throw out the last coordinates
        normal = SVector{NDIMS}(cartesian2spherical(normal..., getindex.(xyzf, idM))[1:NDIMS])
        # normal = global2contravariant(normal, auxM, equations)
        
        flux_face_values[idM] = surface_flux(uM, uP_transformed_to_M, auxM, auxM, normal, equations) * Jf[idM]
    end
end

function Trixi.prolong2interfaces!(cache, u, mesh::DGMultiMesh, 
                                   equations::AbstractCovariantEquations,
                                   surface_integral, dg::DGMulti)
    rd = dg.basis
    @unpack u_values, u_face_values, aux_values, aux_face_values = cache

    Trixi.apply_to_each_field(Trixi.mul_by!(rd.Vf), u_face_values, u)
end


# function Trixi.invert_jacobian!(du, mesh::DGMultiMesh{NDIMS, <:Trixi.NonAffine}, 
#                                 equations::AbstractCovariantEquations,
#                                 dg::DGMulti, cache; scaling = -1) where {NDIMS}
#     Trixi.@threaded for e in eachelement(mesh, dg, cache)
#         invJ = cache.invJ[1, e]
#         aux_node = cache.aux_values[1, e]
#         factor = scaling / area_element(aux_node, equations)
#         for i in axes(du, 1)
#             du[i, e] *= factor
#         end
#     end
# end