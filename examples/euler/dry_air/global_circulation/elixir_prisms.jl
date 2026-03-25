###############################################################################
# DGSEM for the linear advection equation on a prismed icosahedral grid
###############################################################################

using OrdinaryDiffEqLowStorageRK, Trixi, TrixiAtmo

###############################################################################
# Spatial discretization

initial_condition = initial_condition_barotropic_instability

equations = CovariantShallowWaterEquations2D(EARTH_GRAVITATIONAL_ACCELERATION,
                                             EARTH_ROTATION_RATE,
                                             global_coordinate_system = GlobalCartesianCoordinates())

###############################################################################
# Build DG solver.

polydeg = (1, 1)

dg = DGMulti(element_type = Wedge(),
             approximation_type = Polynomial(),
             surface_flux = flux_lax_friedrichs,
             polydeg = polydeg)

###############################################################################
# Build mesh.

horizontal_initial_refinement= 0

mesh = DGMultiMeshPrismIcosahedron3D(dg, 0.5 * EARTH_RADIUS, EARTH_RADIUS;
                                     horizontal_initial_refinement = horizontal_initial_refinement,
                                     vertical_layers = 1)

using StartUpDG: export_to_vtk
md = mesh.md
(; x, y) = md
u = @. (x + y)^2
v = @. x^2 + y^2
vtu_name = export_to_vtk(dg.basis, md, [u, v], "uv.vtu", equi_dist_nodes=false)                                   

