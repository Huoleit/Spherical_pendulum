using TrajectoryOptimization
using MeshCatMechanisms
using RigidBodyDynamics
using MeshCat
using MeshCatMechanisms
using StaticArrays
using Rotations
using LinearAlgebra
using ForwardDiff

using Altro
using RobotDynamics
using PyPlot

struct quadrotor{C} <: LieGroupModel
    mech::Mechanism{Float64}
    statecache::C
    dyncache::DynamicsResultCache{Float64}
    xdot::Vector{Float64}
    function quadrotor(mech::Mechanism)
        N = num_positions(mech) + num_velocities(mech)
        statecache = StateCache(mech)
        rescache = DynamicsResultCache(mech)
        xdot = zeros(N)
        new{typeof(statecache)}(mech, statecache, rescache, xdot)
    end
end

RobotDynamics.LieState(::quadrotor) = RobotDynamics.QuatState(17, SA[1])
RobotDynamics.control_dim(::quadrotor) = 4


function RobotDynamics.dynamics(model::quadrotor, x::AbstractVector{T1}, u::AbstractVector{T2}) where {T1, T2}
    T = promote(T1, T2)
    state = model.statecache[T1]
    res = model.dyncache[T1]

    copyto!(state, x)
    F = [0., 0., u[1]+u[2]+u[3]+u[4]]
    τ = [0.45*(u[2]-u[4]), 0.45*(u[3]-u[1]), (u[1]-u[2]+u[3]-u[4])]
    # τ = u[1:3]
    # F = u[4:6]
    control = [τ;F;0.;0.]
    RigidBodyDynamics.dynamics!(res, state, control)
    q̇ = res.q̇
    v̇ = res.v̇
    return [q̇; v̇]
end

quad_state(model::quadrotor) = [configuration(model.statecache[Float64]);velocity(model.statecache[Float64])]


function initialize_visualizer(a1::quadrotor)
    vis = Visualizer()
    delete!(vis)
    cd(joinpath(@__DIR__,"quad"))
    mvis = MechanismVisualizer(a1.mech, URDFVisuals(URDFPATH), vis)
    cd(@__DIR__)
    return mvis
end

const URDFPATH = joinpath(@__DIR__, "quad","quad.urdf")
quad_mechanism = parse_urdf(URDFPATH, floating=true, remove_fixed_tree_joints=false)
quad = quadrotor(quad_mechanism)

tf = 5.0
Δt = 1e-2
ts = range(0, tf, step=Δt)
N = Int(round(tf/Δt)) + 1

n = num_positions(quad.mech) + num_velocities(quad.mech)
m = 4
Q = 1.0e-2*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0e-2*Diagonal(@SVector ones(m))
x0 = [normalize([1.,0.,0.,0.]);[0.,0.,1.];zeros(10)]
xf = [normalize([1.,0.,0.,0.]);[4.,4.,4.];zeros(10)]
obj = LQRObjective(Q,R,Qf,xf,N)

zero!(quad.statecache[Float64])
hover_force = dynamics_bias(quad.statecache[Float64])[6]
u0 = fill(hover_force/m, m)
U0 = [u0 for k = 1:N-1]
conSet = ConstraintList(n,m,N)
x_bnd = [fill(Inf, 7);0.1;0.1;fill(Inf, 8)]
bnd = BoundConstraint(n, m, x_min=-x_bnd, x_max=x_bnd)
goal = GoalConstraint(xf)
waypoint = GoalConstraint([[0.9330127, 0.25, 0.25, 0.0669873]; zeros(13)], 1:4)
add_constraint!(conSet, bnd, 1:N-1)
add_constraint!(conSet, waypoint, 200)
add_constraint!(conSet, goal, N)

prob = Problem(quad, obj, xf, tf, x0=x0, constraints=conSet)
initial_controls!(prob, U0);
rollout!(prob)
opts = SolverOptions(
    cost_tolerance_intermediate=1e-2,
    penalty_scaling=10.,
    penalty_initial=1.0
)
altro = ALTROSolver(prob, opts);
solve!(altro);


mvis = initialize_visualizer(quad)
render(mvis)
# final_time = 5.
# zero!(quad.statecache[Float64])
# ts, qs, vs = simulate(quad.statecache[Float64], 5.0, Δt = 1e-3);

# MeshCatMechanisms.animate(mvis, ts, qs; realtimerate = 1.);
X = states(altro)
x_traj = [x[1:9] for x in X]
animation = Animation(mvis, ts, x_traj)
setanimation!(mvis, animation)

pygui(true)
theta_traj = [rad2deg.(x[8:9]') for x in X]
theta_traj = vcat(theta_traj...)
plot(ts, theta_traj)

ori_traj = [rad2deg(RotXYZ(UnitQuaternion(x[1:4]...))[1]) for x in X]
ori_traj = vcat(ori_traj...)
plot(ts, ori_traj)


# set_configuration!(quad.statecache[Float64], findjoint(quad.mech, "base_to_world"), [1, 2., 3., 4., .1, .0, .0])
# set_configuration!(mvis, configuration(quad.statecache[Float64]))
# quad.statecache[Float64]