""" Today in this session we are going to generate data for variation in moonlander for
the analytical solution and solve the approximate function using the Machine methods """;

begin
            using NeuralPDE, ModelingToolkit, Flux, DiffEqFlux, GalacticOptim
            using Symbolics
            import ModelingToolkit: infimum, supremum, Interval
            using DomainSets, Quadrature
            using Plots
            using QuasiMonteCarlo, Distributions
            using Quadrature, Cubature
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
4. train the weights and params for the cost function.
5. select the cross validation set to check the accuracy of model if, its high bias model or high variance model.
6. If the model is high bias, we need to add more features. For high variance we need to add more training data.

=#


h_data  = (vec(h_mat))
v_data  = (vec(v_mat))
u_data  = (vec(u_mat))
## SORTING OUR DATA H0 AND V0 AS INPUT AND U AS OUTPUT
using Random
h_data  = shuffle(vec(h_mat))
v_data  = shuffle(vec(v_mat))
u_data  = 1/3*shuffle(vec(u_mat))

ip = [h_data' ; v_data']
op = hcat(u_data...)


## INITIALIZING THE NUMBER OF NEURAL NETWORK LAYERS, HIDDEN UNITS, TRAIN-TEST SETS,
n = 32;
#model = Chain(Dense(2,n,σ),Dense(n,n,relu),Dense(n,n,relu),Dense(n,1))
#model = Chain(Dense(2,n,relu),Dense(n,n,relu),Dense(n,1))
model = Dense(2,1,σ)
# Now we shuffle the data and take 70% of training and 30% test sets


train_no = 1:Int64.(0.7*length(h_data))
test_no = length(train_no) :length(h_data)

train_data, op_train_data = hcat([ip[:,i] for i in train_no]...) , hcat([op[i] for i in train_no]...)
test_data , op_test_data = hcat([ip[:,i] for i in test_no]...) , hcat([op[i] for i in test_no]...)

""" Now use the data and get the things done.......""";
##
function loss(a,b)
      #model(train_data) - out_train_data
      Flux.Losses.mse(model(a),b)
end
loss(train_data,op_train_data)
opt = ADAM(0.01);
data = Iterators.repeated((train_data,op_train_data),1000)
function cb!()
      println("loss of the function is :" ,loss(train_data,op_train_data) )
      println("loss of test data: ", loss(test_data,op_test_data))
end
Flux.train!(loss,Flux.params(model),data,opt,cb =  cb!) #Flux.throttle(cb!,0.05))
pred = vec(model(test_data))
act = vec(op_test_data)

##

plot(vec(model(train_data)),label = "train data")

scatter(pred, label = "predicted data")
scatter!(act, label = "actual data")

scatter(act, label = "actual data")
#=
using Gadfly
plot(x = h_data, y= v_data, color = u_data)
