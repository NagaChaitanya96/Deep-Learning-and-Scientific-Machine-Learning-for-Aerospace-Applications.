cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DifferentialEquations, DiffEqFlux ,Plots, GalacticOptim
# solve
# uₜ+uₓ=0, x,t in (0,1) .
# IC u(x,t=0) = sin(2pi*x)
# BC u(x=0,t) = u(x=1,t)# y=[x,t]

# Defining u′=f(x,t); loss function = mse(uₜ+uₓ≈0)
dudx = FastChain(FastDense(1,20,tanh),FastDense(20,1))
#p=initial_params(dudx)
#u₀(x)=sin(2pi*x)
n=30
xspan=(0.0f0,1.0f0)
xsteps=range(xspan[1],xspan[2],length=n)
uₓ=NeuralODE(dudx,xspan,Tsit5(),saveat=xsteps)
u0=Float32[0.0]

# This is the predicted value for assumed neuralode which varies with x
function predict_neuralode(p)
    Array(uₓ(u0,p))
end
# now we need to match this prediction to the initial condition. This can be done by specifying the loss function as
# MSE predicted function - actual loss value

# predict_neuralode(p)

# actual Initial condition values
function dode(dc,u,p,x)
  dc .= 2*pi*cos(2*pi*x)
end
prob_dode = ODEProblem(dode,u0,xspan)
ode_data = Array(solve(prob_dode,Tsit5(),saveat = xsteps))



data=Array(Float32.([sin(2pi*x) for x in xsteps]))
act_data = reshape(data,1,length(data))

function lossInSpaceDim(p)
    pred = predict_neuralode(p)
    loss = sum(abs2,pred.-act_data)
    return loss ,  pred
end
uₓ.p

# lossInSpaceDim(p)
callback = function (p, l, pred; doplot = true)
  display(l)
  #plt = scatter(xsteps,act_data[1,:], label = "data")
  #scatter!(plt, xsteps,pred[1,:], label = "prediction")
  #if doplot
  #  display(plot(plt))
  #end
  return false
end


res = DiffEqFlux.sciml_train(lossInSpaceDim,uₓ.p,cb=callback)
# final parameters for reduced loss functionj
# here we did not get the exact parameters because the function is stuck at local minima
# however lets continue to define another loss function

p1=res.minimizer
data
pred_data = (predict_neuralode(p1))'
plt1 = plot(xsteps,data)
plt2 = plot(xsteps,pred_data)
plot!(plt1,plt2)
lossInSpaceDim(p1)

# now after successful completino we will know the neuralODE which gives us the parameters p1 for which we get u(0,x) =sin(2πx).
# now we need to define a function such that uₜ+uₓ = 0 satisfies from t= 0.0 to 1.0 i.e u is also a f(t)
# if we assume uₜ as a neural ODE in time steps (t) such that , can we solve the ODE problem which includes a neuralODE

# since  uₜ = -uₓ, tried defining u_τ as an ode problem  which varies with time
f3(du,p,t) = uₓ(u0,p1)
u₀=0.25
tspan = (0.0 ,1.0 )
prob= ODEProblem(f3,u₀,tspan)
sol = solve(prob)



#function PDE_equation(t,x)
#    uₜ,uₓ=Flux.gradient((x,t)->model,t,x)
#    L1=uₜ+uₓ
#    return L1
#end
