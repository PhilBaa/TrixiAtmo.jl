###############################################################################
# DGSEM for the linear advection equation on a prismed icosahedral grid
###############################################################################

using OrdinaryDiffEq, Trixi, TrixiAtmo

###############################################################################
# Spatial discretization

@inline function initial_condition_smooth_mountain(x, t, equations)
    RealT = eltype(x)
    a = sqrt(x[1]^2 + x[2]^2 + x[3]^2)  # radius of the sphere
    lat = asin(x[3] / a)
    h_0 = 5960.0f0
    v_0 = 20.0f0

    # compute zonal and meridional components of the velocity
    vlon, vlat = v_0 * cos(lat), zero(eltype(x))

    # compute geopotential height 
    h = h_0 -
        1 / EARTH_GRAVITATIONAL_ACCELERATION *
        (a * EARTH_ROTATION_RATE * v_0 + 0.5f0 * v_0^2) * (sin(lat))^2

    # Convert primitive variables from spherical coordinates to the chosen global 
    # coordinate system, which depends on the equation type
    return TrixiAtmo.spherical2global(SVector(h, vlon, vlat, zero(RealT),
                                    bottom_topography_smooth_mountain(x)), x,
                            equations)
end

# Bottom topography function to pass as auxiliary_field keyword argument in constructor for 
# SemidiscretizationHyperbolic, used with initial_condition_smooth_mountain
@inline function bottom_topography_smooth_mountain(x)
    RealT = eltype(x)
    a = sqrt(x[1]^2 + x[2]^2 + x[3]^2)  # radius of the sphere
    lon, lat = atan(x[2], x[1]), asin(x[3] / a)

    # Position and height of mountain, noting that latitude is λ = -π/2 and not λ = 3π/2 
    # because atan(y,x) is in [-π, π]
    lon_0, lat_0 = convert(RealT, -π / 2), convert(RealT, π / 6)
    b_0 = 2000.0f0

    R = convert(RealT, π / 9)
    return b_0 * exp(-((lon - lon_0)^2 + (lat - lat_0)^2) / R^2)
end

initial_condition = initial_condition_smooth_mountain

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
                                    auxiliary_field = bottom_topography_smooth_mountain)

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