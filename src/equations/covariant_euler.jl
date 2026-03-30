@muladd begin
#! format: noindent

@doc raw"""
    CovariantEulerEquations{GlobalCoordinateSystem} <:  
        AbstractCovariantEquations{3, 3, GlobalCoordinateSystem, 5}


## References
- Comparison of two Euler equation sets in a Discontinuous Galerkin
solver for atmospheric modelling (BRIDGE v0.9)
Michael Baldauf and Florian Prill
"""
struct CovariantEulerEquation{GlobalCoordinateSystem, RealT <: Real} <:
       AbstractCovariantEulerEquations{3, 3, GlobalCoordinateSystem, 6}
    gravity::RealT  # acceleration due to gravity
    rotation_rate::RealT  # rotation rate for Coriolis term 
    gamma::RealT  # 
    inv_gamma_minus_one::RealT  # 
    global_coordinate_system::GlobalCoordinateSystem
    function CovariantEulerEquations(gravity::RealT,
                                     rotation_rate::RealT,
                                     gamma::RealT;
                                     global_coordinate_system = GlobalCartesianCoordinates()) where {RealT <:
                                                                                                              Real}
        return new{typeof(global_coordinate_system), RealT}(gravity, rotation_rate, gamma, inv(gamma - 1))
    end
end

# TODO: idk
have_nonconservative_terms(::CovariantEulerEquation) = False()

# The conservative variables are the height and contravariant momentum components
function varnames(::typeof(cons2cons), ::AbstractCovariantEulerEquations)
    return ("rho", "rho_vcon1", "rho_vcon2", "rho_vcon3", "rho_e", "phi")
end

# The primitive variables are the height and contravariant velocity components
function varnames(::typeof(cons2prim), ::AbstractCovariantEulerEquations)
    return ("rho", "vcon1", "vcon2", "vcon3", "e", "phi")
end

# The change of variables contravariant2global converts the two local contravariant vector 
# components u[2] and u[3] to the three global vector components specified by 
# equations.global_coordinate_system (e.g. spherical or Cartesian). This transformation 
# works for both primitive and conservative variables, although varnames refers 
# specifically to transformations from conservative variables.
function varnames(::typeof(contravariant2global),
                  ::AbstractCovariantEulerEquations)
    return ("rho", "v1", "v2", "v3", "e", "phi")
end

# Convenience functions to extract physical variables from state vector
@inline density(u, ::AbstractEulerEquations) = u[1]

@inline velocity_contravariant(u,
::AbstractCovariantEulerEquations) = SVector(u[2] /
                                                      u[1],
                                                      u[3] /
                                                      u[1],
                                                        u[4] /
                                                        u[1])
@inline momentum_contravariant(u,
::AbstractCovariantEulerEquations) = SVector(u[2],
                                                      u[3],
                                                        u[4])

@inline total_energy(u, ::AbstractEulerEquations) = u[5] / u[1]

@inline energy_density(u, ::AbstractEulerEquations) = u[5]

@inline potential(u, ::AbstractEulerEquations) = u[6]

@inline function pressure(u, equations::AbstractCovariantEulerEquations)
    rho = density(u, equations)
    vcon = velocity_contravariant(u, equations)
    Gcov = metric_covariant(aux_vars, equations)
    ekin = 0.5f0 * dot(Gcov * vcon, vcon) * rho
    return (equations.gamma - 1) * (total_energy(u, equations) - ekin - potential(u, equations))
end

@inline function cons2prim(u, aux_vars,
                           equations::AbstractCovariantEulerEquations)
    rho, rho_vcon1, rho_vcon2, rho_vcon3, rho_e_total, phi = u
    M = momentum_contravariant(u, equations)
    Gcov = metric_covariant(aux_vars, equations)
    ekin = 0.5f0 * dot(Gcov * M, M) / rho

    p = (equations.gamma - 1) *
        (rho_e_total - ekin - rho * phi)
    return SVector(rho, rho_vcon1 / rho, rho_vcon2 / rho, rho_vcon3 / rho, p, phi)
end

@inline function prim2cons(u, aux_vars,
                           equations::AbstractCovariantEulerEquations)
    rho, vcon1, vcon2, vcon3, p, phi = u
    vcon = SVector(vcon1, vcon2, vcon3)
    Gcov = metric_covariant(aux_vars, equations)
    ekin = 0.5f0 * dot(Gcov * vcon, vcon) * rho
    e_total = ekin + p / (equations.gamma - 1) + rho * phi
    return SVector(rho, rho * vcon1, rho * vcon2, rho * vcon3, rho * e_total, phi)
end

# Entropy variables are w = (g(h+hₛ) - (v₁v¹ + v₂v²)/2, v₁, v₂)ᵀ
@inline function cons2entropy(u, aux_vars,
                              equations::AbstractCovariantEquations)
    h = waterheight(u, equations)
    h_s = bottom_topography(aux_vars, equations)
    vcon = velocity_contravariant(u, equations)
    vcov = metric_covariant(aux_vars, equations) * vcon
    return SVector{3}(equations.gravity * (h + h_s) - 0.5f0 * dot(vcov, vcon),
                      vcov[1], vcov[2])
end

@inline function cons2entropy(u, aux_vars,
                              equations::AbstractCovariantEulerEquations)
    rho, rho_vcon1, rho_vcon2, rho_vcon3, rho_e_total, phi = u

    # Contravariant velocity components
    vcon = velocity_contravariant(u, equations)

    # Covariant components via the metric
    vcov = metric_covariant(aux_vars, equations) * vcon

    # Kinetic energy in covariant form v_i v^i
    v_square = dot(vcov, vcon)

    # Pressure (remove gravitational potential contribution from total energy)
    p = (equations.gamma - 1) * (rho_e_total - 0.5f0 * rho * v_square - rho * phi)

    # Thermodynamic entropy and rho/p
    s = log(p) - equations.gamma * log(rho)
    rho_p = rho / p

    # Entropy variables (covariant form)
    w1 = (equations.gamma - s) * equations.inv_gamma_minus_one -
         0.5f0 * rho_p * v_square
    w2 = rho_p * vcov[1]
    w3 = rho_p * vcov[2]
    w4 = rho_p * vcov[3]
    w5 = -rho_p

    return SVector(w1, w2, w3, w4, w5, zero(eltype(u)))
end

# Convert contravariant momentum components to the global coordinate system
@inline function contravariant2global(u, aux_vars,
                                      equations::AbstractCovariantEulerEquations)
    rho_v1, rho_v2, rho_v3 = basis_covariant(aux_vars, equations) *
                             momentum_contravariant(u, equations)
    return SVector(u[1], rho_v1, rho_v2, rho_v3, u[5], u[6])
end

# Convert momentum components in the global coordinate system to contravariant components
@inline function global2contravariant(u, aux_vars,
                                      equations::AbstractCovariantEulerEquations)
    rho_vcon1, rho_vcon2, rho_vcon3 = basis_contravariant(aux_vars, equations) *
                       SVector(u[2], u[3], u[4])
    return SVector(u[1], rho_vcon1, rho_vcon2, rho_vcon3, u[5], u[6])
end

# Entropy function (total energy) given by S = rho * e_total - rho * phi - 0.5 * rho * v_i v^i
@inline function entropy(u, aux_vars,
                         equations::AbstractCovariantEulerEquations)
    rho, rho_vcon1, rho_vcon2, rho_vcon3, rho_e_total, phi = u

    # Contravariant velocity components
    vcon = velocity_contravariant(u, equations)

    # Covariant components via the metric
    vcov = metric_covariant(aux_vars, equations) * vcon

    # Kinetic energy in covariant form v_i v^i
    v_square = dot(vcov, vcon)

    # Pressure (remove gravitational potential contribution from total energy)
    p = (equations.gamma - 1) * (rho_e_total - 0.5f0 * rho * v_square - rho * phi)

    # Thermodynamic entropy and rho/p
    s = log(p) - equations.gamma * log(rho)
    rho_p = rho / p

    # Entropy variables (covariant form)
    w1 = (equations.gamma - s) * equations.inv_gamma_minus_one -
         0.5f0 * rho_p * v_square
    w2 = rho_p * vcov[1]
    w3 = rho_p * vcov[2]
    w4 = rho_p * vcov[3]
    w5 = -rho_p

    return SVector(w1, w2, w3, w4, w5, zero(eltype(u)))
end

# Flux as a function of the state vector u, as well as the auxiliary variables aux_vars, 
# which contain the geometric information required for the covariant form
@inline function flux(u, aux_vars, orientation::Integer,
                      equations::CovariantEulerEquations)
    # Geometric variables
    Gcon = metric_contravariant(aux_vars, equations)
    J = area_element(aux_vars, equations)

    # Physical variables
    rho_vcon = momentum_contravariant(u, equations)
    vcon = velocity_contravariant(u, equations)
    rho_e_total = energy_density(u, equations)

    # Compute and store the pressure and Energy momentum tensor components in the desired orientation
    p = pressure(u, equations)
    T = rho_vcon * vcon[orientation] + p * Gcon[:, orientation]

    return SVector(J * rho_vcon[orientation], J * T..., J * vcon[orientation] * (rho_e_total + p), zero(eltype(u)))

end

# Flux as a function of the state vector u, as well as the auxiliary variables aux_vars, 
# which contain the geometric information required for the covariant form
@inline function flux(u, aux_vars, normal_direction::AbstractVector,
                      equations::CovariantEulerEquations)
    # Geometric variables
    Gcon = metric_contravariant(aux_vars, equations)
    J = area_element(aux_vars, equations)

    # Physical variables
    rho = density(u, equations)
    rho_vcon = momentum_contravariant(u, equations)
    rho_e_total = energy_density(u, equations)

    # Compute and store the pressure and Energy momentum tensor components in the desired direction
    vcon = dot(rho_vcon, normal_direction) / rho
    p = pressure(u, equations)
    T = rho_vcon * vcon + p * (Gcon * normal_direction)

    return SVector(J * dot(rho_vcon, normal_direction), J * T..., J * vcon * (rho_e_total + p), zero(eltype(u)))
end

# Standard geometric and Coriolis source terms for a rotating sphere
@inline function source_terms_geometric_coriolis(u, x, t, aux_vars,
                                                 equations::CovariantEulerEquations)
    error("Source terms for the full 3D Euler equations are not yet implemented")
end

# Maximum wave speed along the normal direction in reference space
@inline function max_abs_speed(u_ll, u_rr, aux_vars_ll, aux_vars_rr,
                               orientation::Integer,
                               equations::CovariantEulerEquations)
    error("max_abs_speed for the full 3D Euler equations is not yet implemented")
end

# Maximum wave speed along the normal direction in reference space
@inline function max_abs_speed(u_ll, u_rr, aux_vars_ll, aux_vars_rr,
                               normal_direction::AbstractVector,
                               equations::CovariantEulerEquations)
    error("max_abs_speed for the full 3D Euler equations is not yet implemented")
end

# Maximum wave speeds with respect to the covariant basis
@inline function max_abs_speeds(u, aux_vars,
                                equations::CovariantEulerEquations)
    error("max_abs_speeds for the full 3D Euler equations is not yet implemented")
end

# If the initial velocity field is defined in Cartesian coordinates and the chosen global 
# coordinate system is spherical, perform the appropriate conversion
@inline function cartesian2global(u, x,
                                  ::CovariantEulerEquations{GlobalSphericalCoordinates})
    h_vlon, h_vlat, h_vrad = cartesian2spherical(u[2], u[3], u[4], x)
    return SVector(u[1], h_vlon, h_vlat, h_vrad, u[5], u[6])
end

# If the initial velocity field is defined in spherical coordinates and the chosen global 
# coordinate system is Cartesian, perform the appropriate conversion
@inline function spherical2global(u, x,
                                  ::CovariantEulerEquations{GlobalCartesianCoordinates})
    h_vx, h_vy, h_vz = spherical2cartesian(u[2], u[3], u[4], x)
    return SVector(u[1], h_vx, h_vy, h_vz, u[5], u[6])
end

# If the initial velocity field is defined in spherical coordinates and the chosen global 
# coordinate system is spherical, do not convert
@inline function spherical2global(u, x,
                                  ::CovariantEulerEquations{GlobalSphericalCoordinates})
    return u
end
end # @muladd
