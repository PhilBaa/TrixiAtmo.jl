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

# version for affine meshes
function Trixi.calc_volume_integral!(du, u, mesh::DGMultiMesh{NDIMS_AMBIENT},
                               have_nonconservative_terms::False,
                               equations::AbstractCovariantEquations{NDIMS},
                               volume_integral::VolumeIntegralWeakForm, dg::DGMulti,
                               cache) where {NDIMS_AMBIENT, NDIMS}
    rd = dg.basis
    (; aux_quad_values) = cache
    @unpack weak_differentiation_matrices, dxidxhatj, u_values, local_values_threaded = cache
    @unpack rstxyzJ, J = mesh.md
    rd = dg.basis
    (; Vq) = rd

    Jq = Vq * J

    # interpolate to quadrature points
    Trixi.apply_to_each_field(Trixi.mul_by!(rd.Vq), u_values, u)

    Trixi.@threaded for e in Trixi.eachelement(mesh, dg, cache)
        flux_values = local_values_threaded[Threads.threadid()]
        for i in 1:NDIMS    # specialization for 2D covariant equations
            for j in Trixi.eachindex(flux_values)
                u_node = u_values[j, e]
                aux_node = aux_quad_values[j, e]
                detg = area_element(aux_node, equations)
                J_node = Jq[j, e]
                flux_values[j] = flux(u_node, aux_node, i, equations) * (J_node / detg)
                
            end

            Trixi.apply_to_each_field(Trixi.mul_by_accum!(weak_differentiation_matrices[i]),
                                    view(du, :, e), flux_values)
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
        # TODO: Is there no function to transform vectors to the reference element?
        normal = global2contravariant((0, normal...), auxM, equations)[2:end]

        
        detg = area_element(auxM, equations)

        flux_face_values[idM] = surface_flux(uM, uP_transformed_to_M, auxM, auxM, normal, equations) / detg * Jf[idM] 
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