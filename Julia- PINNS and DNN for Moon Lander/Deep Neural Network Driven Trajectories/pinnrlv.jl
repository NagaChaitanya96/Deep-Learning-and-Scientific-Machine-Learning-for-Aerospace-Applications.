using NeuralPDE, DiffEqFlux, Plots, GalacticOptim, Flux
using DifferentialEquations
using ModelingToolkit, Quadrature, Cubature
import ModelingToolkit: Interval, infimum, supremum
##
""" Loading initial conditions and data """;
Cd=0.5;
S_ref=10.75;
Isp=300;
I_z=3346393;
l_com=9.778;# % centre of mass distance
t_ref=32;
R0=6378*10^3;# % Earth's radius
g0=9.81;
S_ref=10.75;
Cd=0.5;
Isp=300;
I_z=3346393;
l_com=9.778; #% centre of mass distance
t_ref=32;
h0=5000;     #   % No variation accepted %5000
s0=850;       #  % 850 and  855 are only working for this downrange distance %850
v0=325;        # % acceptable range 325 -351 %325
gamma0 = -75*pi/180;# %initial 75 changing gamma0 and theta0 in equal ranges gives good results
# % whereas compared to individual variation in both the initial angles.
m0=26229.667;
theta0 = -75*pi/180; #% initial 75
omega0 = 0.342;
c3 = (g0*t_ref/v0)
T′max = 756.222e3/(m0*g0);
# Dr=0.5*1.225*((y(3)*v0).^2).*Cd*S_ref/(m0*g0);
##
""" """;
@parameters t′,p1 ,p2
# r , s , v , γ, m , θ , ω
@variables u1(..),u2(..),u3(..),u4(..),u5(..),u6(..),u7(..),u8(..),u9(..)
Dt = Differential(t′)
# D = 0.5*1.225*((u1(3)*v0).^2)*Cd*S_ref/(m0*g0)
eqs = [Dt(u1(t′)) ~ u3(t′)*sin(u4(t′)),
        Dt(u2(t′)) ~ u3(t′) * cos(u4(t′)),
        Dt(u3(t′)) ~ c3* ((u8(t′) * cos(u9(t′) - u4(t′) + u6(t′)) + (0.5*1.225*((u1(3)*v0).^2)*Cd*S_ref/(m0*g0)))/u5(t′) + sin(u4(t′))/(u1(t′))^2),
        Dt(u4(t′)) ~ c3* ((u8(t′) * sin(u9(t′) - u4(t′) + u6(t′)))/ (u5(t′)*u3(t′)) + cos(u4(t′))/ ((u1(t′))^2*u3(t′))),
        Dt(u5(t′)) ~ -t_ref * u8(t′)/Isp,
        Dt(u6(t′)) ~ u7(t′) * t_ref,
        Dt(u7(t′)) ~ -u8(t′) * sin(u9(t′)) *(t_ref*m0*g0*l_com)/I_z,
        Dt(u8(t′)) ~ 25 - p1^2,
        Dt(u9(t′)) ~ t_ref*(4π/180 -p2^2)]

bcs = [u1(0) ~ (R0+5000)/R0,
        u1(1) ~ 1,
        u2(0) ~ 0,
        u2(1) ~ 1,
        u3(0) ~ 1,
        u3(1) ~ 5/v0,
        u4(0) ~ -75π/180,
        u4(1) ~ -π/2,
        u5(0) ~ 1,
        u5(1) ~ 21296.10/m0,
        u6(0) ~ -75π/180,
        u6(1) ~ -π/2,
        u7(0) ~ 0.342,
        u7(1) ~ 0,
        u8(0) ~ T′max,
        u8(1) ~ 0.5*T′max,
        u9(0) ~ -10π/180,
        u9(1) ~ 0]

domain = [t′ ∈ Interval(0.0,1.0)]

n =16## Neural NEtwork
chain = [FastChain(FastDense(1,n,Flux.σ),FastDense(n,n,relu),FastDense(n,n,relu),FastDense(n,1,Flux.σ)) for _ in 1:9]

##
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
dt = 0.01
_strategy = NeuralPDE.GridTraining(dt)
discretization = PhysicsInformedNN(chain, _strategy, init_params= initθ, param_estim = true)

@named pde_system = PDESystem(eqs,bcs,domain,[t′],[u1(t′),u2(t′),u3(t′),u4(t′),u5(t′),u6(t′),u7(t′),u8(t′),u9(t′)], [p1,p2], defaults = Dict([p .=>1.0 for p in [p1,p2]]))
prob = discretize(pde_system,discretization)
sym_prob = symbolic_discretize(pde_system,discretization)

pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
bcs_inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents

callback = function (p,l)
    println("loss: ", l )
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end

res = GalacticOptim.solve(prob,ADAM(0.01); callback = callback, maxiters=50)

phi = discretization.phi

phi[0]([0,  ])
# u_predict  = [[phi[i]([t′],minimizers_[i])[1] for t in ts  for x in xs] for i in 1:3]
