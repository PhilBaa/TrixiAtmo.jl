###############################################################################
# DGSEM for the linear advection equation on a prismed icosahedral grid
###############################################################################

using OrdinaryDiffEq, Trixi, TrixiAtmo

###############################################################################
# Spatial discretization

initial_condition = initial_condition_geostrophic_balance

equations = CovariantShallowWaterEquations2D(EARTH_GRAVITATIONAL_ACCELERATION,
                                                  EARTH_ROTATION_RATE,
                                                  global_coordinate_system = GlobalCartesianCoordinates())

###############################################################################
# Build DG solver.

tensor_polydeg = (2, 1)

dg = DGMulti(element_type = Wedge(),
             approximation_type = Polynomial(),
             surface_flux = flux_central,
             polydeg = tensor_polydeg)


volume_flux = flux_central
# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
dg = DGMulti(polydeg = tensor_polydeg, element_type = Wedge(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

###############################################################################
# Build mesh.

mesh = DGMultiMeshPrismIcosahedron(dg;
    inner_radius = 0.99 * EARTH_RADIUS,
    outer_radius = EARTH_RADIUS,
    initial_refinement = 3)

# Transform the initial condition to the proper set of conservative variables
initial_condition_transformed = transform_initial_condition(initial_condition, equations)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_transformed, dg)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0 to T
ode = semidiscretize(semi, (0.0, 12 * SECONDS_PER_DAY))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation 
# setup and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the 
# results
analysis_callback = AnalysisCallback(semi, interval = 10,
                                     save_analysis = true,
                                     extra_analysis_errors = (:conservation_error,),
                                     uEltype = real(dg))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 1,
                                     solution_variables = contravariant2global)

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
            dt = 1.0, save_everystep = true, callback = callbacks)
