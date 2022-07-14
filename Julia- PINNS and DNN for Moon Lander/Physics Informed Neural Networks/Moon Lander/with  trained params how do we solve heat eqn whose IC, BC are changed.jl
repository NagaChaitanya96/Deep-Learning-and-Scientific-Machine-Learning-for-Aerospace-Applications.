""" Packages usage """;
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

begin
        @parameters t,x;
        @variables u(..);

        Dx = Differential(x);
        Dt = Differential(t);
        Dxx = Differential(x).^2;

        eqn = Dt(u(t,x)) ~ Dxx(u(t,x));
        bcs = [u(0,x) ~ cos(x),
                u(t,0) ~ exp(-t),
                u(t,1) ~ exp(-t)* cos(1)];

        domains = [x ∈ Interval(0.0,1.0),
                        t ∈ Interval(0.0,1.0)];
end
##
""" Defining NeuralNetwork """;;
dx = 0.05
begin
        Hu =50 #hiddenunits =16;
        layers = 4;
        chain = FastChain(FastDense(2,Hu ,σ),FastDense(Hu ,Hu ,σ),FastDense(Hu ,Hu ,σ),FastDense(Hu ,Hu ,σ),FastDense(Hu,1,σ));
        initθ = Float64.(DiffEqFlux.initial_params(chain))
        discretization = PhysicsInformedNN(chain,GridTraining(dx);inital_params = initθ);
        indvars = [t,x];
        depvars = [u(t,x)];
end
""" Discretize and Optimize""";
##
begin
        @named pdesys = PDESystem(eqn,bcs,domains,indvars,depvars);
        prob = discretize(pdesys,discretization);
        cb_ = function (p,l)
            #println("loss: ", l , "losses: ", map(l -> l(p), lossfunctions
            println("the current loss is : $l")
            return false
        end;
        opt1 = BFGS();opt2 = ADAM(0.001);
        maxiters=500;
        res = GalacticOptim.solve(prob, opt1; cb = cb_, maxiters=maxiters)
        phi = discretization.phi
end

##
ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
u_predict = hcat([[first(phi([t,x],res.minimizer)) for x in xs] for t in ts]...)
u_ex = (t,x) -> exp.(-t) * cos.(x);
u_exact = [u_ex(t,x) for x in xs , t ∈ ts]
# now creating a function which can predict the values for different xs values
function norms(ts,xs,f,g)
        l2 = sum([((f .- g).^2)[:,i] for i ∈ 1: length(ts)])./(length(xs))
        l2percent = sum([(((u_predict.-u_exact)./u_exact).^2)[:,i] for i in 1 :length(xs)])
        l∞ = maximum([(abs.(f .- g))[:,i] for i in 1: length(ts)])
        return l2, l∞
end
l2norm,linfinity = norms(ts,xs,u_predict,u_exact)
# the accuracy obtained using this solve is the current loss is : 0.00020193108693132258;  now when we use the parameters onto the new equation
# can we expect the accuracy of at least 1 order less than what we have obtained previously for the same number of iterations performed on same
# optimization algorithm?

# the current loss is : 0.0004201337139820842
##
""" Now we change the equations""";
eqn = Dt(u(t,x)) ~ Dxx(u(t,x));
bcs = [u(0,x) ~ 1.1*cos(x),
        u(t,0) ~ exp(-t),
        u(t,1) ~ exp(-t)* cos(1)];
#uexact_new = [[1.1*cos(x)*exp(-t) for x in xs ] for t in ts]
initθ = Float64.(res.minimizer)
discretization1 = PhysicsInformedNN(chain,GridTraining(dx);initial_params = initθ)
@named pdesys1 = PDESystem(eqn,bcs,domains,indvars,depvars);
prob2 = discretize(pdesys1,discretization1)
cb1 = function (p,l)
        println("the loss for the new equation using the old params is $(l)")
        return false
end
res = GalacticOptim.solve(prob2,BFGS(),cb=cb1,maxiters = 500)

# the loss for the new equation using the old params is 0.0041192964456282614

upredict_new = hcat(firstphi([t,x],res.minimizer) for x in xs] for t in ts]...)
l2norm1,linfinity1 = norms(ts,xs,upredict_new,uexact_new)

##
  """ CONCLUSION""";
#=
This way of solving the problem will not have better params since for every single change in the equation we will be having change in the parameters. So even if we
give some trained values for other equation it might not happen that our params will effect the number of iterations of the solution.

1. Instead we can try using training the model with lower grid size and updating the gridsize as we attain better accuracy, in this manner we will have a
   rough estimate of our solution.

   example: if we train our model to accuracy of 1e-5 using the stepsize of 0.1 we will have a good estimate of parameters. and we can use them directly for dx =0.001
            this will lead to converging the solution directly from at least 1e-3, which is much better than solving from a higher loss value.
