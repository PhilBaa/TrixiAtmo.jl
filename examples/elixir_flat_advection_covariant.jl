###############################################################################
# DGSEM for the linear advection equation on a prismed icosahedral grid
###############################################################################
# To run a convergence test, use
# convergence_test("../examples/elixir_spherical_advection_covariant_quad_icosahedron.jl", 4, cells_per_dimension = (1,1))

using OrdinaryDiffEq, Trixi, TrixiAtmo

###############################################################################
# Spatial discretization

cells_per_dimension = (20, 20, 2)
initial_condition = initial_condition_flat

equations = CovariantLinearAdvectionEquation2D(global_coordinate_system = GlobalCartesianCoordinates(), flat = true)

###############################################################################
# Build DG solver.

tensor_polydeg = (1, 1) # culprit for DomainError

dg = DGMulti(element_type = Wedge(),
             approximation_type = Polynomial(),
             surface_flux = flux_central,
             polydeg = tensor_polydeg)

###############################################################################
# Build mesh.

mesh = DGMultiMesh(dg,
                   cells_per_dimension,
                   coordinates_min = (-1.0, -1.0, -1.0),
                   coordinates_max = (1.0, 1.0, 1.0),
                   periodicity = true)



# Transform the initial condition to the proper set of conservative variables
initial_condition_transformed = transform_initial_condition(initial_condition, equations)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_transformed, dg)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0 to T
ode = semidiscretize(semi, (0.0, 5.0))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation 
# setup and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the 
# results
#analysis_callback = AnalysisCallback(semi, interval = 10,
#                                     save_analysis = true,
#                                     extra_analysis_errors = (:conservation_error,))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 1,
                                     solution_variables = contravariant2global)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.7)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE 
# solver
callbacks = CallbackSet(save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed 
# callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, save_everystep = true, callback = callbacks)
