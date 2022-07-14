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
""" For different values of h we get matrix of h,u,v w.r.t time and velocity
h = 9.95 -> h0 in h_dist[1]
v = -2.05 -> v0 in v_dist[1]

1. Now perturbing from 11*11 in H and V we will be obtaining different values, without converting
      to vector we will have 40*11*11 matrix.

2. Now our data has repeated points and corresponding values.

3. We will sort our data into ascending order and seperate out the repeated data, to make sure our Neural Network
      parameters will be consistent without any excess weightage given to the repeated points.

4. Now for each data point variation we are calculating the results for moon lander for 100 time steps
""";

h_mat = [];
v_mat = [];
u_mat = [];
t_all = [];

for v0 in v_dist
      for h0 in h_dist
                  tfinal = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
                  h_data = [AnalyticalSol(a,v0,h0)[4] for a in LinRange(0,tfinal,5)]
                  v_data = [AnalyticalSol(a,v0,h0)[3] for a in LinRange(0,tfinal,5)]
                  u_data = [AnalyticalSol(a,v0,h0)[5] for a in LinRange(0,tfinal,5)]

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
t_all == vec(tf_dist') # this states that for each combination of v0,h0 we have enough data and that is true.

##
""" Now that we have the necessary data we choose the package to get the approximate neural network trained """;
using Flux
#=
Now what is the input and output?

We need to train a NN such that our params will be able to predict minimum fuel consumption at given states, since our
objective function is minimizing the U.

1. Define a neural network, with Hidden Units and Layers
2. divide the set of data to train and test sets.     (randomize the dataset)
3. define a cost function which should be minimized
. train the weights and params for the cost function.
5. select the cross validation set to check the accuracy of model if, its high bias model or high variance model.
6. If the model is high bias, we need to add more features. For high variance we need to add more training data.

=#

## SORTING OUR DATA H0 AND V0 AS INPUT AND U AS OUTPUT
h_data  = vec(h_mat)
v_data  = vec(v_mat)
u_data  = vec(u_mat)
ip = [h_data' ; v_data']
op = hcat(u_data...)
using Random
data2 = [h_data v_data u_data]
data3 = hcat([collect(data2[j,:])  for j in randperm(length(op))]...) # shuffled data for training
X = data3[1:2,:]
Y = 1/3*hcat(data3[3,:]...)


## INITIALIZING THE NUMBER OF NEURAL NETWORK LAYERS, HIDDEN UNITS, TRAIN-TEST SETS,
#model = Chain(Dense(2,n,σ),Dense(n,n,relu),Dense(n,n,relu),Dense(n,1))
# Now we shuffle the data and take 70% of training and 30% test sets

n = 32;
#model = Chain(Dense(2,n,σ),Dense(n,n,relu),Dense(n,1,relu))
model = Chain(Dense(2,n,relu),Dense(n,n,relu),Dense(n,n,relu),Dense(n,1,σ))
function loss(a,b)
      Flux.Losses.mse(model(a),b)
end
ss = Int64.(0.8*length(Y))
x1 , y1 = X[:,1:ss] , Y[:,1:ss]
xt, yt = X[:,ss+1:end] , Y[:,ss+1: end]

opt = ADAM(0.01);
data = Iterators.repeated((x1,y1),500)
err_train= [];
err_test= [];

function cb!()
      push!(err_train,loss(x1,y1))
      push!(err_test, loss(xt,yt))
      println("loss of the function is :" ,loss(x1,y1))
end
#Flux.train!(loss,Flux.params(model),data,opt,cb = cb!)
#Flux.train!(loss,Flux.params(model),Iterators.repeated((training_data_X,training_data_Y),500),opt,cb = cb!)
@time Flux.train!(loss,Flux.params(model),data,opt,cb = cb!)
pred = 3*vec(model(xt))
yt_a = 3*yt
plot(pred,label = "predicted")
plot!(vec(yt_a),label = "actual")
function L2(a,b)
      return (sum((a-b).^2))/(length(a))
end
L2(pred,vec(yt_a))
#Flux.train!(loss,Flux.params(model),Iterators.repeated((X,Y),500),ADAM(0.005),cb = cb!)
plot(log10.(err_train))
plot!(log10.(err_test))

#=

=#
