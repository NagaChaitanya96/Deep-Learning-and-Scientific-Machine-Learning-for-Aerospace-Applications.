begin
    using NeuralPDE, ModelingToolkit, DiffEqFlux, GalacticOptim, Plots
    using DifferentialEquations
    using DiffEqOperators
    using DomainSets
    using OrdinaryDiffEq
    using CUDA
end
##
begin
    """ Symbolic representation""";
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
    dx = 0.1       # grid training;
end

##
begin
    """ Defining NeuralNetwork """;;
    Hu =16 #hiddenunits =16;
    layers = 3;

    chain = FastChain(FastDense(2,Hu ,σ),FastDense(Hu ,Hu ,σ),FastDense(Hu ,Hu ,σ),FastDense(Hu,1,σ));
    initθ = Float64.(DiffEqFlux.initial_params(chain)) |>gpu;
    discretization = PhysicsInformedNN(chain,GridTraining(dx);inital_params = initθ);
    indvars = [t,x];
    depvars = [u(t,x)];
    @named pdesys = PDESystem(eqn,bcs,domains,indvars,depvars);
    prob = discretize(pdesys,discretization);
end


cb_ = function (p,l)
    #println("loss: ", l , "losses: ", map(l -> l(p), lossfunctions))
    println("the current loss is : $l")
    return false
end;

begin
    """ solving the optimization problem""";
    opt1 = BFGS(); opt2 = ADAM(0.005);
    maxiters= 1000;
    res = GalacticOptim.solve(prob, opt2; cb = cb_, maxiters=maxiters);
    prob = remake(prob,u0= res.minimizer)
    res = GalacticOptim.solve(prob, opt1; cb = cb_, maxiters=maxiters);
    prob = remake(prob,u0= res.minimizer)
    res = GalacticOptim.solve(prob, opt2; cb = cb_, maxiters=maxiters);
    phi = discretization.phi
end

ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
u_ex = (t,x) -> exp.(-t) * cos.(x);

begin  """ solutions """;
    u_predict = reshape([first(phi([t,x],res.minimizer)) for t in ts for x in xs ],(length(xs),length(ts)))
    u_exact = reshape([u_ex(t,x) for t in ts for x in xs], (length(xs),length(ts)))
end


function norms(ts,xs,f,g)
        l2 = sum([((f .- g).^2)[:,i] for i ∈ 1: length(ts)])./(length(xs))
        l2percent = sum([(((u_predict.-u_exact)./u_exact).^2)[:,i] for i in 1 :length(xs)])
        l∞ = maximum([(abs.(f .- g))[:,i] for i in 1: length(ts)])
        return l2, l∞, l2percent
end

l2norm,linfinity,l2percent = norms(ts,xs,u_predict,u_exact)

begin
        l2norm = sum([((u_predict .- u_exact).^2)[:,i] for i ∈ 1: size(u_predict)[1]])./(size(u_predict))[2]

        p1 = plot(xs,log10.(l2norm),title = "L2 norm using NPDE",label =" ADAM +BFGS +ADAM
        each Iter =$(maxiters)" , xticks = 0.0:0.25:1.0)
        xlabel!("Distance X")
        ylabel!("log10(Error)")
        #png("D:\\1e-4 to 1e-6 accuracy is attained 3")

        linfinity = maximum([(abs.(u_predict .- u_exact))[:,i] for i in size(u_predict)[1]])./size(u_predict)[2]
        p2 = plot(xs,log10.(linfinity),title = "L∞ norm using NPDE",xticks = 0.0:0.25:1.0)
        xlabel!("Distance X")
        ylabel!("log10(Error) ")

        #png("D:\\1e-4 to 1e-6 accuracy is attained 4")
        plot(p1,p2)
end

#plot(p1,p2)
#png("D:\\l2 vs linf for 15000 iterations")
