
"""
MOON LANDER with taking U as a neural network and assuming the constraints to be
                  U(0) ~ 0. and u(tf) ~ 3.
We make a prediction of the Analytical Function - POLY, SIGMOID,  without the analytical data
and simply solve the problem based on the physical constraints and check whether we reach the
correct solution

""";

##
begin
      using ModelingToolkit
      using DiffEqFlux
      using NeuralPDE
      import ModelingToolkit: Interval
      using GalacticOptim
end


"""Since we cant solve ode and also estimate parameters just by giving tspan and initial conditions, let us
give the final conditions also and verify whether the paraemeters and minimum values obtained are constant""";
begin
      @parameters t,a,b
      @variables h(..),v(..), u(..)

      Dt = Differential(t)
      #  H, V are +ve upward direction , since flight is descending we have v0 = -2 as initial condition, and h0 =10.0 initially. As the aircraft reaches
      # the ground the Hfinal should be 0 and v final should reach 0. For this we need to minimize the energy consumption ∫Udt.

      # to solve this we are solving the system of ODE's and given the initial conditions. As a part of ModelingToolkit interface, we give the indvars and
      # depvars.

      """ parameters obtained for getting analytical udata are  a = 157, b = 221 """
      eqn = [Dt(h(t)) ~ v(t),
                  Dt(v(t)) ~ -1.5 + 3*1/(1+exp(-a*t+b)) ,
                  Dt(u(t)) ~ 3*a*(exp(-a*t+b)/(1+exp(-a*t+b))^2)]


      v0= -2.; h0 = 10.
      tf = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))

      begin
            bcs = [h(0.) ~ 10.,
                    v(0.) ~ -2., # its decelerating
                    u(0.) ~ 0.,
                    u(tf) ~ 3.,
                    h(tf) ~ 0.,
                    v(tf) ~ 0.,
                    ]
      end

      domains = [t in Interval(0.,tf)]
      indvars = [t]
      depvars = [h(t), v(t), u(t)]
      #depvars = [h(t), v(t)]
end

# Function for Analytical Solution {{{
function AnalyticalSol(t,v0,h0)
      tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
      ŝ = tf̂/2 + v0/3
      v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
      ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
      û(t) = (t<=ŝ) ? 0 : 3
      #println("The values are : $([tf̂, ŝ ,v̂(t) , ĥ(t), û(t)])")
      return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
end

begin
      cb = function(p,l)
              println("The loss is :",l)
              return false
      end
      """ We wanted to reduce the Thrust value  i.e  J:= ∫udt so for U we need to get the parameters of a,b from each
      iteration in the res.minimizer""";
end

begin
      parameters = [a,b]
      initGuess = Dict([p => 10. for p in parameters])
      # Check to see if there is an ODE System function
      @named pde_system = PDESystem(eqn,bcs,domains,indvars,depvars,parameters,defaults = initGuess)
      n = 10
      chain = [FastChain(FastDense(1 ,n,σ),FastDense(n,n,σ),FastDense(n,1)) for _ in 1:3]
      dt = 0.01;
      ts = 0.:dt:tf
      t_ = collect(ts)
      u21 = zeros(length(t_))
      t2 = t_
      grid_strategy = NeuralPDE.GridTraining(dt);
      discretization = PhysicsInformedNN(chain,grid_strategy,param_estim = true, additional_loss = nothing)
end

begin
      initθ = discretization.init_params
      acum =  [0;accumulate(+, length.(initθ))]
      sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
      phi = discretization.phi
end


begin
      addloss(phi,θ,p) = Hpen(phi,θ,p)
      prob = NeuralPDE.discretize(pde_system,discretization)

      res = GalacticOptim.solve(prob, BFGS();cb =cb,maxiters = 1000)

      res.minimizer[end-1:end]
      initθ = discretization.init_params;
      acum =  [0;accumulate(+, length.(initθ))];
      sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1];
      phi = discretization.phi
      minimizers = [res.minimizer[s] for s in sep]
      u_predict = [[first(phi[i](t,minimizers[i])) for t in ts] for i in 1:3]
end

begin
      using Plots
      hdata = [AnalyticalSol(a,-2.,10.)[4] for a in ts]
      udata =  [AnalyticalSol(a,-2.,10.)[5] for a in ts]
      vdata = [AnalyticalSol(a,-2.,10.)[3] for a in ts]
      u_ = [hdata'; vdata'; udata']
      len = length(t_)

      p11 = plot(ts,[u_[i,:] for i in 1:3],title = " exact solution ",yticks = -4.:1.:10,xticks = 0.0 :0.5: 4.2)
      p12 = plot(ts, u_predict, label = ["h(t)" "v(t)" "u(t)"],xticks = 0.0 :0.5: 4.2,yticks = -4.:1.:10, title = "NeuralPDE solution")
      println("parameters are:",res.minimizer[end-1:end])
      println("minimum value of h,v, vmax", minimum(u_predict[1]) ,  minimum(u_predict[2]), maximum(u_predict[2]))
      plot(p11,p12)

end
png("D:\\Chaitanya\\sciml\\Analysis so far\\ Moonlander without data and estimating the hyper parameters 1 ")


#=
""" Plotting graph for Thrust only """;
begin
      p = res.minimizer[end-1:end]
      Uvec =  [3/(1+exp(-p[1]*t+p[2])) for t in t_]
      plot(title = "variation of predicted vs actual thrust")
      plot!(t_,udata, xticks = 0.0 :0.5:4.2, yticks = 0.0 :0.5:3.5, label ="Actual thrust" )
      plot!(t_,Uvec, xticks = 0.0 :0.5:4.2, yticks = 0.0 :0.5:3.5; label = "Thrust prediction")
end


"""
Conclusion:
      1. For any initial guess of a and b we will be able to get the approximate curve.
      2. If assuming C also as a parameter which will be multiplied with sigmoid function, we need a good initial guess
         to start with else, we end up with inaccurate approximation. However
"""
