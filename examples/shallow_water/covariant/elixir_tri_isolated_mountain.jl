###############################################################################
# DGSEM for the linear advection equation on a prismed icosahedral grid
###############################################################################

using OrdinaryDiffEq, Trixi, TrixiAtmo
using ForwardDiff

###############################################################################
# Spatial discretization

initial_condition = initial_condition_isolated_mountain

equations = CovariantShallowWaterEquations2D(EARTH_GRAVITATIONAL_ACCELERATION,
                                             EARTH_ROTATION_RATE,
                                             global_coordinate_system = GlobalCartesianCoordinates())

###############################################################################
# Build DG solver.

polydeg = 4

dg = DGMulti(element_type = Tri(),
             approximation_type = Polynomial(),
             surface_flux = flux_lax_friedrichs,
             polydeg = polydeg)


###############################################################################
# Build mesh.

initial_refinement_level = 4

mesh = DGMultiMeshTriIcosahedron2D(dg, EARTH_RADIUS;
                                   initial_refinement = initial_refinement_level)

# Transform the initial condition to the proper set of conservative variables
initial_condition_transformed = transform_initial_condition(initial_condition, equations)

# Standard geometric and Coriolis source terms for a rotating sphere
@inline function source_terms_geometric_coriolis_bottom_topography(u, x, t, aux_vars,
                                                                   equations::CovariantShallowWaterEquations2D)
    # Geometric variables
    Gcon = TrixiAtmo.metric_contravariant(aux_vars, equations)
    Gamma1, Gamma2 = TrixiAtmo.christoffel_symbols(aux_vars, equations)
    J = TrixiAtmo.area_element(aux_vars, equations)

    # Physical variables
    h = waterheight(u, equations)
    h_vcon = TrixiAtmo.momentum_contravariant(u, equations)
    v_con = TrixiAtmo.velocity_contravariant(u, equations)

    # Doubly-contravariant flux tensor
    momentum_flux = h_vcon * v_con' + 0.5f0 * equations.gravity * h^2 * Gcon

    # Coriolis parameter
    f = 2 * equations.rotation_rate * x[3] / sqrt(x[1]^2 + x[2]^2 + x[3]^2)  # 2Ωsinθ

    # Geometric source term
    s_geo = SVector(sum(Gamma1 .* momentum_flux), sum(Gamma2 .* momentum_flux))

    # Combined source terms
    source_1 = s_geo[1] + f * J * (Gcon[1, 2] * h_vcon[1] - Gcon[1, 1] * h_vcon[2])
    source_2 = s_geo[2] + f * J * (Gcon[2, 2] * h_vcon[1] - Gcon[2, 1] * h_vcon[2])

    grad_bottom_topography = TrixiAtmo.bottom_topography_derivatives(aux_vars, equations)

    source_1 += equations.gravity * h * (Gcon[1, 1] * grad_bottom_topography[1] + Gcon[1, 2] * grad_bottom_topography[2])
    source_2 += equations.gravity * h * (Gcon[2, 1] * grad_bottom_topography[1] + Gcon[2, 2] * grad_bottom_topography[2])

    # Do not scale by Jacobian since apply_jacobian! is called before this
    return SVector(zero(eltype(u)), -source_1, -source_2)
end

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_transformed, dg,
                                    source_terms = source_terms_geometric_coriolis_bottom_topography,
                                    auxiliary_field = bottom_topography_isolated_mountain)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0 to T
ode = semidiscretize(semi, (0.0, 15 * SECONDS_PER_DAY))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation 
# setup and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the 
# results
analysis_callback = AnalysisCallback(semi, interval = 100,
                                     save_analysis = true,
                                     extra_analysis_errors = (:conservation_error,),
                                     uEltype = real(dg))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 300,
                                     solution_variables = cons2prim_and_vorticity)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.7)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE 
# solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed 
# callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, save_everystep = false, callback = callbacks)