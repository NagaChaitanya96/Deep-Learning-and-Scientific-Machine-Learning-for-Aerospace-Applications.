""" Packages usage """;
begin
        using NeuralPDE, ModelingToolkit, DiffEqFlux, GalacticOptim, Plots
        using DiffEqOperators
        using DifferentialEquations
        using DomainSets
        using OrdinaryDiffEq
        using CUDA
end
dx = 0.05     # grid training;
begin
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
                """ Defining NeuralNetwork """;
                begin
                        Hu =22 #hiddenunits =16;
                        layers = 2;
                        chain = FastChain(FastDense(2,Hu ,σ),FastDense(Hu ,Hu ,σ),FastDense(Hu,1,σ));
                        initθ = Float64.(DiffEqFlux.initial_params(chain)) |>gpu;
                        #initθ = Float64.(res.minimizer)
                        discretization = PhysicsInformedNN(chain,GridTraining(dx);inital_params = initθ);
                        indvars = [t,x];
                        depvars = [u(t,x)];
                end
                """ Discretize and Optimize""";
                begin
                        @named pdesys = PDESystem(eqn,bcs,domains,indvars,depvars);
                        prob = discretize(pdesys,discretization);
                        cb_ = function (p,l)
                            #println("loss: ", l , "losses: ", map(l -> l(p), lossfunctions))
                            println("the current loss is : $l")
                            return false
                        end;
                        opt1 = BFGS();opt2 = ADAM(0.01);
                        maxiters=2000;
                        res = GalacticOptim.solve(prob, opt2; cb = cb_, maxiters=maxiters)
                        phi = discretization.phi
                end
end

ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
u_predict = hcat([[first(phi([t,x],res.minimizer)) for x in xs] for t in ts]...)

u_ex = (t,x) -> exp.(-t) * cos.(x);
u_exact = [u_ex(t,x) for x in xs , t ∈ ts]
# now creating a function which can predict the values for different xs values
u_predict1(ts,xs) = hcat([[first(phi([t,x],res.minimizer)) for x in xs] for t in ts]...)
u_exact1(ts,xs) = [u_ex(t,x) for x in xs , t ∈ ts]

# Generating points in between the samples
function gp(dx) #generate points
        points = collect(0:dx:1.0)
        points[2:length(points)-1] = points[3:length(points)] .- (dx/2)* rand(1)
        return points
        end


function norms(ts,xs,f,g)
        l2 = sum([((f .- g).^2)[:,i] for i ∈ 1: length(ts)])./(length(xs))
        l2percent = sum([(((u_predict.-u_exact)./u_exact).^2)[:,i] for i in 1 :length(xs)])
        l∞ = maximum([(abs.(f .- g))[:,i] for i in 1: length(ts)])
        return l2, l∞, l2percent
end
l2norm,linfinity,l2percent = norms(ts,xs,u_predict,u_exact)

""" log10 PLOTS""";

theme(:bright)
p1 = plot()
plot!(title = "L2 Norm for random points generated
        inside the domain" , xticks = 0.0:0.25:1.0,
                        xlabel = "time in (s)",ylabel = "log10(ϵ)", xlims = (0.0,1.0), ylims = (-16.0,0.0),
                        yticks = 0.0:-1.0:-16.0, frame =:box,label = "at δx = $(dx)")
scatter!(ts,log10.(norms(ts,xs,u_predict,u_exact)[1]), label = "At exact stepsize ")
i = 1
while i< 6
        xs1 = gp(dx)
        display(plot!(ts,log10.(norms(ts,xs1,u_predict1(ts,xs1),u_exact1(ts,xs1))[1]),label = "randomly generated points"))
        i=i+1
end

p2 = plot()
plot!(title = "L∞ Norm for random points generated
        inside the domain" , xticks = 0.0:0.25:1.0,
                        xlabel = "time in (s)",ylabel = "log10(ϵ)", xlims = (0.0,1.0), ylims = (-11.0,0.0),
                        yticks = 0.0:-1.0:-16.0, frame =:box,label = "at δx = $(dx)")
scatter!(ts,log10.(norms(ts,xs,u_predict,u_exact)[2]), label = "At exact stepsize ")
i = 1
while i< 6
        xs1 = gp(dx)
        display(plot!(ts,log10.(norms(ts,xs1,u_predict1(ts,xs1),u_exact1(ts,xs1))[2]),label = "randomly generated points"))
        i=i+1
end



theme(:bright)
p3 = plot()
plot!(title = "L2percent for random points generated
        inside the domain" , xticks = 0.0:0.25:1.0,
                        xlabel = "time in (s)",ylabel = "log10(ϵ)", xlims = (0.0,1.0), ylims = (-16.0,0.0),
                        yticks = 0.0:-1.0:-16.0, frame =:box,label = "at δx = $(dx)")
scatter!(ts,log10.(norms(ts,xs,u_predict,u_exact)[3]), label = "At exact stepsize ")
i = 1
while i< 6
        xs1 = gp(dx)
        display(plot!(ts,log10.(norms(ts,xs1,u_predict1(ts,xs1),u_exact1(ts,xs1))[3]),label = "randomly generated points"))
        i=i+1
end


initial_guess = Float64.(res.minimizer)



# Now we want to create sample points randomly between each of the given dx spacing to check
# whether we will be able to get the accuracy while using the parameters at 0.05.

for i in 1:21
        println((sum(u_predict[:,i])/21))
end

f2 = [(u_predict - u_exact)[:,i] for i in 1 : 21]
maximum_values_index = findfirst(isequal(maximum(f2)),f2)


""" FINDING THE MAXIMUM VALUE INDEX  """;


""" Error """;
plot()
plot!([(u_predict - u_exact)[:,i] for i in 1 : 3:21])
xlabel!("x")
ylabel!("upred -uexact not absolute")
#png("D:\\2021\\Final year Project\\sciml\\November\\Nov2\\error variation1")

#=
plot()
plot!([abs.(u_predict - u_exact)[:,i] for i in 1 : 3:21])
xlabel!("x")
ylabel!("upred -uexact not absolute")
png("D:\\2021\\Final year Project\\sciml\\November\\Nov2\\absolute error variation3")


plot()
plot!([((u_predict .- u_exact).^2)[:,i] for i in 1:3:21])
xlabel!("x")
ylabel!("ϵ^2")
png("D:\\2021\\Final year Project\\sciml\\November\\Nov2\\abs2 error variation4 ")
=#
