"""  VARIATION IN LEARNING RATE.
""";
cd(@__DIR__)
begin
           using NeuralPDE, ModelingToolkit, Flux, DiffEqFlux, GalacticOptim
           using Symbolics
           import ModelingToolkit: infimum, supremum, Interval
           using DomainSets, Quadrature
           using Plots
           using QuasiMonteCarlo, Distributions
           using Quadrature, Cubature
           using BenchmarkTools
end

     ##

     """ Analytical solution details""";
     v0= -2.; h0 = 10.
     tf = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
     function AnalyticalSol(t,v0,h0)
          tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
          ŝ = tf̂/2 + v0/3
          v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
          ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
          û(t) = (t<=ŝ) ? 0 : 3
          return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
     end

##
""" Now vary the details by 5% in v0 and h0 in 11 steps
V,H disturbance = [v0,h0] ±5% """;

dist = (-5:1:5)*0.01
v_dist,h_dist = v0.+collect(dist), h0.+collect(dist)
# v_dist will be available in the rows and h_dist will be available in columns
tf_dist = [(2/3)*(v1)+ (4/3*(sqrt((v1^2/2) + 1.5*h1))) for v1 in v_dist, h1 in h_dist]

##
h_matrix = [];
u_matrix = [];
v_matrix = [];
h_mat = [];
v_mat = [];
u_mat = [];
t_all = [];

for v0 in v_dist
     for h0 in h_dist
                 tfinal = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
                 h_data = [AnalyticalSol(a,v0,h0)[4] for a in LinRange(0,tfinal,100)]
                 v_data = [AnalyticalSol(a,v0,h0)[3] for a in LinRange(0,tfinal,100)]
                 u_data = [AnalyticalSol(a,v0,h0)[5] for a in LinRange(0,tfinal,100)]

                 h_matrix = push!(h_matrix,h_data)
                 v_matrix = push!(v_matrix,v_data)
                 u_matrix = push!(u_matrix,u_data)

                 h_mat = hcat(h_matrix...)
                 v_mat = hcat(v_matrix...)
                 u_mat = hcat(u_matrix...)
                 t_all = push!(t_all,tfinal)
     end
end

""" To check we have the data for all the v0 and h0 at all times we choose to verify the time """;

##
""" Now that we have the necessary data we choose the package to get the approximate neural network trained """;
using Flux
h_data  = vec(h_mat)
v_data  = vec(v_mat)
u_data  = vec(u_mat)
ip = [h_data' ; v_data']
op = hcat(u_data...)
""" DONT USE SHUFFLE , BECAUSE IT SHUFFLES ALL THE DATA IN ENTIRE MATRIX.""";
using Random
ss = Int64.(0.85*length(u_data))
rng = MersenneTwister(1234)
""" Shuffling done here""";
numbers = shuffle(rng,Vector(1:length(u_data)))
data2 = [h_data v_data u_data]
data2 = [data2[i,:] for i in eachrow(numbers)]
data2 = reduce(vcat,data2)
##
""" Setting to  test and train sets """;
X = data2[:,[1,2]]'
Y = data2[:,3]'
x1 , y1 = X[:,1:ss] , Y[:,1:ss]
xt, yt = X[:,ss+1:end] , Y[:,ss+1: end]
##
n = 128;
model = Chain(Dense(2,n,σ),Dense(n,n,relu),Dense(n,n,relu),Dense(n,1,relu))
#model = Chain(Dense(2,n,leakyrelu),Dense(n,n,leakyrelu),Dense(n,n,leakyrelu),Dense(n,1,relu))
function loss(a,b)
     Flux.Losses.mse(model(a),b)
end


opt = ADAM(0.005);
data = Iterators.repeated((x1,y1),1000)
ps = Flux.params(model)

err_train= [];
err_test= [];
function cb!()
     push!(err_train,loss(x1,y1))
     push!(err_test, loss(xt,yt))
     println("loss of the function is :" ,loss(x1,y1))
end
#Flux.train!(loss,Flux.params(model),data,opt,cb = cb!)
#Flux.train!(loss,Flux.params(model),Iterators.repeated((training_data_X,training_data_Y),500),opt,cb = cb!)
#@time Flux.train!(loss,Flux.params(model),Iterators.repeated((x1,y1),3000),ADAM(0.05),cb = cb!)
@time Flux.train!(loss,Flux.params(model),data,opt,cb = cb!)
#@time Flux.train!(loss,Flux.params(model),data,ADAM(0.005),cb = cb!)

##

#Flux.train!(loss,Flux.params(model),Iterators.repeated((X,Y),500),ADAM(0.005),cb = cb!)
pred = vec(model(xt))
plot(pred,label = "predicted")
plot!(vec(yt),label = "actual")


plot(log10.(err_train))
plot!(log10.(err_test))
ps = Flux.params(model)


hnew=[];
vnew=[];
unew=[];
##
""" Now we are going to the evaluate the how the values of h,v are obtained for the values of u that
we send into the Euler integration. So for the analytical solution of h0,v0 ,we have a trained model
which will send the values of u, for our initial conditions.

Now using the u values as our initial guess and at each time step we are running the Euler SOlver
forward in time so that, we will estimate the desired h and v .

""";

""" Now to make sure we have the right accuracy we test our model using the EUler integration strategy""";
v0= -1.95; h0 = 10.5
tfinal  = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))

h = [AnalyticalSol(a,v0,h0)[4] for a in LinRange(0,tfinal,1000)]
v = [AnalyticalSol(a,v0,h0)[3] for a in LinRange(0,tfinal,1000)]

u = model([h';v'])
hnew =[];
vnew = [];

"""
SInce I have u values for certain time steps , I am going to use the same u values and check whether
I will be able to get the correct h ,v values
""";
function EulerInt(tfinal,u)
     time1 = LinRange(0,tfinal,1000)
     dt = time1[2]-time1[1]
     hnew = [h0]; vnew = [v0];
     for i in 1:length(time1)-1
           hn = hnew[i] + dt * vnew[i]
           vn = vnew[i] + dt * (-1.5+u[i])
           hnew = push!(hnew,hn)
           vnew = push!(vnew, vn)
     end
     return hnew,vnew
end

hnew,vnew = EulerInt(tfinal,u)
"""
Eulers model gives approximate values for the trained u(θ,X) where X is the states. Even though,
Eulers model is simplest model to perform numerical integration, we are able to achieve approximate
results upto some level. However, the numerical integration performed violates the constraints
so we go for higher order integration procedures like RK4 integration.
"""
##
#=
""" SAVING THE MODEL """;
using BSON: @save, @load
@save "moonLander_model.bson" model
=#
