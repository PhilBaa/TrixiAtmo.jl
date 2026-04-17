@muladd begin
#! format: noindent

@doc raw"""
    CovariantEulerEquations3D{GlobalCoordinateSystem} <:  
        AbstractCovariantEquations{3, 3, GlobalCoordinateSystem, 5}


## References
- Comparison of two Euler equation sets in a Discontinuous Galerkin
solver for atmospheric modelling (BRIDGE v0.9)
Michael Baldauf and Florian Prill
"""
struct CovariantEulerEquations3D{GlobalCoordinateSystem, RealT <: Real} <:
       AbstractCovariantEquations{3, 3, GlobalCoordinateSystem, 5}
    gravity::RealT  # acceleration due to gravity
    rotation_rate::RealT  # rotation rate for Coriolis term 
    gamma::RealT  # 
    inv_gamma_minus_one::RealT  # 
    global_coordinate_system::GlobalCoordinateSystem
    function CovariantEulerEquations3D(gravity::RealT,
                                       rotation_rate::RealT,
                                       gamma::RealT;
                                       global_coordinate_system = GlobalCartesianCoordinates()) where {RealT <:
                                                                                                              Real}
        return new{typeof(global_coordinate_system), RealT}(gravity, rotation_rate, gamma, inv(gamma - 1))
    end
end

# TODO: idk
have_nonconservative_terms(::CovariantEulerEquations3D) = False()

# The conservative variables are the height and contravariant momentum components
function varnames(::typeof(cons2cons), ::CovariantEulerEquations3D)
    return ("rho", "rho_vcon1", "rho_vcon2", "rho_vcon3", "rho_e")
end

# The primitive variables are the height and contravariant velocity components
function varnames(::typeof(cons2prim), ::CovariantEulerEquations3D)
    return ("rho", "vcon1", "vcon2", "vcon3", "e")
end

# The change of variables contravariant2global converts the two local contravariant vector 
# components u[2] and u[3] to the three global vector components specified by 
# equations.global_coordinate_system (e.g. spherical or Cartesian). This transformation 
# works for both primitive and conservative variables, although varnames refers 
# specifically to transformations from conservative variables.
function varnames(::typeof(contravariant2global),
                  ::CovariantEulerEquations3D)
    return ("rho", "v1", "v2", "v3", "e")
end

# Convenience functions to extract physical variables from state vector
@inline density(u, ::CovariantEulerEquations3D) = u[1]

@inline velocity_contravariant(u,
::CovariantEulerEquations3D) = SVector(u[2] /
                                                      u[1],
                                                      u[3] /
                                                      u[1],
                                                        u[4] /
                                                        u[1])
@inline momentum_contravariant(u,
::CovariantEulerEquations3D) = SVector(u[2],
                                                      u[3],
                                                        u[4])

@inline total_energy(u, ::CovariantEulerEquations3D) = u[5] / u[1]

@inline energy_density(u, ::CovariantEulerEquations3D) = u[5]

@inline function pressure(u, aux_vars, equations::CovariantEulerEquations3D)
    rho = density(u, equations)
    vcon = velocity_contravariant(u, equations)
    Gcov = metric_covariant(aux_vars, equations)
    ekin = 0.5f0 * dot(Gcov * vcon, vcon) * rho
    phi = geopotential(aux_vars, equations)
    return (equations.gamma - 1) * (total_energy(u, equations) - ekin - rho * phi)
end

@inline function cons2prim(u, aux_vars,
                           equations::CovariantEulerEquations3D)
    rho = density(u, equations)
    vcon1, vcon2, vcon3 = velocity_contravariant(u, equations)
    phi = geopotential(aux_vars, equations)
    p = pressure(u, aux_vars, equations)
    return SVector(rho, vcon1, vcon2, vcon3, p, phi)
end

@inline function prim2cons(u, aux_vars,
                           equations::CovariantEulerEquations3D)
    rho, vcon1, vcon2, vcon3, p = u
    vcon = SVector(vcon1, vcon2, vcon3)
    Gcov = metric_covariant(aux_vars, equations)
    ekin = 0.5f0 * dot(Gcov * vcon, vcon) * rho
    phi = geopotential(aux_vars, equations)
    e_total = ekin + p / (equations.gamma - 1) + rho * phi
    return SVector(rho, rho * vcon1, rho * vcon2, rho * vcon3, rho * e_total)
end

@inline function cons2entropy(u, aux_vars,
                              equations::CovariantEulerEquations3D)
    rho, rho_vcon1, rho_vcon2, rho_vcon3, rho_e_total = u

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

    return SVector(w1, w2, w3, w4, w5)
end

# Convert contravariant momentum components to the global coordinate system
@inline function contravariant2global(u, aux_vars,
                                      equations::CovariantEulerEquations3D)
    rho_v1, rho_v2, rho_v3 = basis_covariant(aux_vars, equations) *
                             momentum_contravariant(u, equations)
    return SVector(u[1], rho_v1, rho_v2, rho_v3, u[5])
end

# Convert momentum components in the global coordinate system to contravariant components
@inline function global2contravariant(u, aux_vars,
                                      equations::CovariantEulerEquations3D)
    rho_vcon1, rho_vcon2, rho_vcon3 = basis_contravariant(aux_vars, equations) *
                       SVector(u[2], u[3], u[4])
    return SVector(u[1], rho_vcon1, rho_vcon2, rho_vcon3, u[5])
end

# Entropy function (total energy) given by S = rho * e_total - rho * phi - 0.5 * rho * v_i v^i
@inline function entropy(u, aux_vars,
                         equations::CovariantEulerEquations3D)
    rho, rho_vcon1, rho_vcon2, rho_vcon3, rho_e_total = u

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

    return SVector(w1, w2, w3, w4, w5)
end

# Flux as a function of the state vector u, as well as the auxiliary variables aux_vars, 
# which contain the geometric information required for the covariant form
@inline function flux(u, aux_vars, orientation::Integer,
                      equations::CovariantEulerEquations3D)
    # Geometric variables
    Gcon = metric_contravariant(aux_vars, equations)
    J = volume_element(aux_vars, equations)

    # Physical variables
    rho_vcon = momentum_contravariant(u, equations)
    vcon = velocity_contravariant(u, equations)
    rho_e_total = energy_density(u, equations)

    # Compute and store the pressure and Energy momentum tensor components in the desired orientation
    p = pressure(u, aux_vars, equations)
    T = rho_vcon * vcon[orientation] + p * Gcon[:, orientation]

    return SVector(J * rho_vcon[orientation], J * T..., J * vcon[orientation] * (rho_e_total + p))

end

# Flux as a function of the state vector u, as well as the auxiliary variables aux_vars, 
# which contain the geometric information required for the covariant form
@inline function flux(u, aux_vars, normal_direction::AbstractVector,
                      equations::CovariantEulerEquations3D)
    # Geometric variables
    Gcon = metric_contravariant(aux_vars, equations)
    J = volume_element(aux_vars, equations)

    # Physical variables
    rho = density(u, equations)
    rho_vcon = momentum_contravariant(u, equations)
    rho_e_total = energy_density(u, equations)

    # Compute and store the pressure and Energy momentum tensor components in the desired direction
    vcon = dot(rho_vcon, normal_direction) / rho
    p = pressure(u, aux_vars, equations)
    T = rho_vcon * vcon + p * (Gcon * normal_direction)

    return SVector(J * dot(rho_vcon, normal_direction), J * T..., J * vcon * (rho_e_total + p))
end

# Standard geometric and Coriolis source terms for a rotating sphere
@inline function source_terms_geometric_coriolis(u, x, t, aux_vars,
                                                 equations::CovariantEulerEquations3D)
    error("Source terms for the full 3D Euler equations are not yet implemented")
end

# Maximum wave speed along the normal direction in reference space
@inline function max_abs_speed(u_ll, u_rr, aux_vars_ll, aux_vars_rr,
                               orientation::Integer,
                               equations::CovariantEulerEquations3D)
    error("max_abs_speed for the full 3D Euler equations is not yet implemented")
end

# Maximum wave speed along the normal direction in reference space
@inline function max_abs_speed(u_ll, u_rr, aux_vars_ll, aux_vars_rr,
                               normal_direction::AbstractVector,
                               equations::CovariantEulerEquations3D)
    rho_ll, v1_ll, v2_ll, v3_ll, p_ll = cons2prim_global(u_ll, aux_vars_ll, equations)
    rho_rr, v1_rr, v2_rr, v3_rr, p_rr = cons2prim_global(u_rr, aux_vars_rr, equations)

    # Calcualte the normal velocities and sound speeds
    v_ll = (v1_ll * normal_direction[1] 
            + v2_ll * normal_direction[2] 
            + v3_ll * normal_direction[3])
    v_rr = (v1_rr * normal_direction[1]
            + v2_rr * normal_direction[2]
            + v3_rr * normal_direction[3])
    c_ll = sqrt(equations.gamma * p_ll / rho_ll)
    c_rr = sqrt(equations.gamma * p_rr / rho_rr)

    norm_ = norm(normal_direction)
    return max(abs(v_ll) + c_ll * norm_, abs(v_rr) + c_rr * norm_)
end

# Maximum wave speeds with respect to the covariant basis
@inline function max_abs_speeds(u, aux_vars,
                                equations::CovariantEulerEquations3D)
    rho, v1, v2, v3, p = cons2prim_global(u, aux_vars, equations)
    c = sqrt(equations.gamma * p / rho)

    return abs(v1) + c, abs(v2) + c, abs(v3) + c
end

# If the initial velocity field is defined in Cartesian coordinates and the chosen global 
# coordinate system is spherical, perform the appropriate conversion
@inline function cartesian2global(u, x,
                                  ::CovariantEulerEquations3D{GlobalSphericalCoordinates})
    h_vlon, h_vlat, h_vrad = cartesian2spherical(u[2], u[3], u[4], x)
    return SVector(u[1], h_vlon, h_vlat, h_vrad, u[5])
end

# If the initial velocity field is defined in spherical coordinates and the chosen global 
# coordinate system is Cartesian, perform the appropriate conversion
@inline function spherical2global(u, x,
                                  ::CovariantEulerEquations3D{GlobalCartesianCoordinates})
    h_vx, h_vy, h_vz = spherical2cartesian(u[2], u[3], u[4], x)
    return SVector(u[1], h_vx, h_vy, h_vz, u[5])
end

# If the initial velocity field is defined in spherical coordinates and the chosen global 
# coordinate system is spherical, do not convert
@inline function spherical2global(u, x,
                                  ::CovariantEulerEquations3D{GlobalSphericalCoordinates})
    return u
end
end # @muladd
