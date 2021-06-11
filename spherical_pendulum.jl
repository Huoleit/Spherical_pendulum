using StaticArrays
using Parameters
using RobotDynamics
using LinearAlgebra

import RobotDynamics: dynamics
import RobotDynamics: state_dim, control_dim

@with_kw struct SphericalPendulum{T} <: AbstractModel
    m::T = 0.3
    length::T = 0.2
    g::T = 9.81
end

function dynamics(p::SphericalPendulum, x, u)
    @unpack m, length, g = p
    θ = x[1]
    ϕ = x[2]
    θ̇ = x[3]
    ϕ̇ = x[4]
    @SMatrix [x[3:4]; ϕ̇^2*cos(θ)*sin(θ)-g/length*sin(θ); -2*ϕ̇*θ̇*cos(θ)/sin(θ)]
end

state_dim(::SphericalPendulum) = 4
control_dim(::SphericalPendulum) = 0
