using ControlSystems
using LinearAlgebra
using RigidBodyDynamics


function RobotDynamics.dynamics(model::quadrotor, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1, T2}
    T = promote(T1, T2)
    state = model.statecache[T1]
    res = model.dyncache[T1]

    copyto!(state, x)
    Ct = 5.84e-06
    Cq = 0.06

    thrust = u.^2 .* Ct
    F = [0., 0., sum(thrust)]
    τ = [0.22*(-thrust[1]+thrust[2]+thrust[3]-thrust[4]), 0.13*(thrust[1]-thrust[2]+thrust[3]-thrust[4]), Cq*(thrust[1]+thrust[2]-thrust[3]-thrust[4])]
    # τ = u[1:3]
    # F = u[4:6]
    control = [τ;F]
    RigidBodyDynamics.dynamics!(res, state, control)
    q̇ = res.q̇
    v̇ = res.v̇
    return [q̇; v̇]
end

quad_state(model::quadrotor) = [configuration(model.statecache[Float64]);velocity(model.statecache[Float64])]


const URDFPATH = joinpath(@__DIR__, "iris","iris.urdf")
quad_mechanism = parse_urdf(URDFPATH, floating=true, remove_fixed_tree_joints=false)
quad = quadrotor(quad_mechanism)
spatial_inertia(findbody(quad_mechanism, "base_link"))
mass(quad_mechanism)

function hat(v)
    return [0 -v[3] v[2];
            v[3] 0 -v[1];
            -v[2] v[1] 0]
end

function L(q)
    s = q[1]
    v = q[2:4]
    L = [s    -v';
            v  s*I+hat(v)]
    return L
end

T = Diagonal([1; -ones(3)])
H = [zeros(1,3); I]

function qtoQ(q)
    return H'*T*L(q)*T*L(q)*H
end

function G(q)
    G = L(q)*H
end

function rptoq(ϕ)
    (1/sqrt(1+ϕ'*ϕ))*[1; ϕ]
end

function qtorp(q)
    q[2:4]/q[1]
end

function E(q)
    E = BlockDiagonal([1.0*I(3), G(q), 1.0*I(6)])
end