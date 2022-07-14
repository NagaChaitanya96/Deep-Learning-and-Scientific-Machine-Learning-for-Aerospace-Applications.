"""
Function to choose the NeuraL Network based on the error approximated
1. hyperparams(η,iterations,HU,Layers,train_test_split,activation functions)
2. define high bias prob and high variance prob using if else statements and
make the code automatically choose the necessary .
3. Initial ranges of each hyper parameter should xXbe given and the model should
be obtained by itself
4. Hypercube should be created such that it randomly selects some set of
4. Check different models and store the model with params having least loss value.
5. Check the model also for robustness, by running the set for 10 times and validate the loss.

""";
##
using Flux,DiffEqFlux, Plots
using BSON: @save, @load
HU_range = [2^(n) for n in 1:9]
Layer_range = [n for n in 1:5]
η_range = [10^r for r in LinRange(-4,-0.31,9)];
##
""" BABY SITTING 1 MODEL """;
using Random
# η = η_range[randperm(9)[1]]*rand()
# n = HU_range[randperm(9)[1]]
# model_vec = [];
train_test_err_vec = [];
train_loss_vec = [];
test_loss_vec = [];
model_vec = [];
best_model_vec = [];
lowest_err_vec = [];

##
# rob = 1;
plot()
function model_training_HYperparam_tuning(η,n,X,Y,ii,iteration3;split = 0.8)
    ip_layer = Dense(3,n,σ)
    op_layer = Dense(n,2)
    hid_layers = Chain(Dense(n,n,relu),Dense(n,n,relu))
    global model = Chain(ip_layer,hid_layers,op_layer)
    # """ Shuffling done here""";
    numbers = randperm(MersenneTwister(1234),Int64(maximum(size(X))))
    # numbers1 = randperm(MersenneTwister(1234),Int64(maximum(size(x))))

    data = [X Y]
    data = [data[i,:] for i in eachrow(numbers)]
    data = reduce(vcat,data)
    # numbers = shuffle(rng,Vector(1:length(u_data)))
    # data2 = [h_data v_data u_data]
    # data2 = [data2[:,i] for i in ]
    """ We are doing only train-test split """;
    ss = Int64(floor(split*maximum(size(X)))) # training set split ends at this index
    # ss2 = ss+1: maximum(size(X)) #dev set split ends at this index
    #ss3 = # test set split
    """ This X & Y will be the shuffled data """;
    X = data[:,[1,2,3]]'
    Y = data[:,[4,5]]'
    """ Train and test sets are divided as follows""";
    x1,y1 = X[:,1:ss], Y[:,1:ss]
    xt,yt = X[:,ss+1:end],Y[:,ss+1:end]
    """ loss function definintion""";
    loss(a,b) = Flux.Losses.mse(model(a),b) #*minimum(size(y)) # now our loss is not the mean error of all
    i1 = 1;
    global err_train
    err_train = [];
    err_test = [];
    function cb!()
        push!(err_train,loss(x1,y1))
        push!(err_test,loss(xt,yt))
        println("current training loss is :", loss(x1,y1))
        # if (i1 % 20) == 0 # for every 20 iterations we are displaying the loss value.
        #     println("current training loss is :", loss(x1,y1))
        #     i1 = i1+1;
        # end
    end
    data2 = Iterators.repeated((x1,y1),iteration3)
    Flux.train!(loss,Flux.params(model),data2,ADAM(η),cb =cb!)
    """ To verify if there is bias of variance between train and test set we check the error
    at the final iteration between train and test set """;
    err_between_train_test = abs(last(err_train) - last(err_test))
    train_loss = abs(last(err_train))
    test_loss = abs(last(err_test))
    push!(train_test_err_vec, err_between_train_test)
    push!(train_loss_vec,train_loss)
    push!(test_loss_vec,test_loss)
    push!(model_vec,model)

    fig1 = plot(title = " Train vs test error for each iteration;
    LR: $(η);HU: $(n)")
    plot!(err_train)
    plot!(err_test)
    display(fig1)
    png(pwd()*"\\Moon Lander\\\\17th march $(η) and $(n) fig $(ii)")
    tol = 1e-2;
    if err_between_train_test > tol
        print("High variance problem reduce number of layers or increase data, dropout layers")
    elseif err_between_train_test ≈ tol
        print("We have good model")
        @save "RLV_good_model.bson" model
    else err_between_train_test < tol
        print("High bias problem, we need to add more layers, should be done manually")
    end
end

function Robust(η,n,X,Y,iteration2,iteration3)
    # this for loop creates a vector which runs for iteration 2 times and includes iteration 3 for tota no of iterations.
    [model_training_HYperparam_tuning(η,n,x,y,ii,iteration3) for ii in 1:iteration2]
    return model_vec,test_loss_vec,train_loss_vec,train_test_err_vec,lowest_err_vec
end
 # size shoudl be 10 as we iterate it 10 times else clear model_vec and all other data

##
function NAS(HU_range,η_range,X,Y,iteration1 , iteration2 , iteration3 ;η=0.5, iterations = 2500,HU = 2,Layers = 3,split = 0.8)
    """ FOR GENERATING DIFFERENT MODELS WITH CHOSEN HYPERPARAMS WE INITIATE RANDOM CHOICE""";
    # iteration1 is responsible for hyperparam_tuning
    # iteration2 for evaluating the robustness for the given hyperparams
    # iteration3 for training our loss function.
    for i in 1: iteration1
        η = η_range[randperm(9)[1]]*rand()
        n = HU_range[randperm(9)[1]]
        display([η n])

        """ ROBUSTNESS OF THE MODEL NEEDS TO BE VERIFIED i.e, with same Learning rate and ARCHItec""";
        """ If model has similar range of error vals, then our model is robust.""";
        Robust(η,n,X,Y,iteration2,iteration3)
        """ the next 4 steps select the best model out of the 10 randomly trained model and send
        to the best_model_vec """;
        global f12 = [train_test_err_vec model_vec]
        min_val = minimum(train_test_err_vec)
        idx_no = reduce(vec,findall(x->x==min_val,f12))[1]
        push!(best_model_vec,model_vec[idx_no])
        push!(lowest_err_vec,train_test_err_vec[idx_no])

    end
    # """
    # Before checking the best model out of random models, we check the Robustness of 1 particular model and
    # verify if it has the minimum train,test,train_test_err and check which model out of the 10 iterations
    # has all the 3 values less and save that model.
    #
    # Then we initialize random search for given hyperparams and so the above step, best of the 10 is saved.
    #
    # """;
    # model_training_HYperparam_tuning(η,n,X,Y)
    # return model

    # hyperparam_tuning = [η_range ,HU_range]
    # η = η_range[randperm(9)[1]]
    # n = HU_range[randperm(9)[1]]
    # model_training_HYperparam_tuning(η,n)

    # """ BABY SITTING 1 MODEL """;

    # test_data = data[:,:][ss+1:ss2]
    return  train_test_err_vec,train_loss_vec,test_loss_vec,model_vec,lowest_err_vec,best_model_vec

end



# f13 = best_model_vec
# min_val = minimum(train_test_err_vec)
# idx_no = findall(x->x=min_val,f12)
# push!(best_model_vec,model_vec[idx_no])

"""
10 * 10 * 2000
1. 10 random hyper param tuning
2. each model run 10 times for robustness.
3. with 2000 iterations for convergence.


Robustness file save all models trained
we find minimum of all the trained models and save 1 best model for that particular η and n.
similarly we do it for 10 times. which means out of 10*10  models we get 10 best models and we
need to choose 1 model out of all.
""";

##
# train_test_err_vec,train_loss_vec,test_loss_vec,model_vec =
x,y = rand(50,3), rand(50,2)
iteration1 = iteration2 =1
iteration3 = 60
train_test_err_vec,train_loss_vec,test_loss_vec,model_vec,lowest_err_vec,best_model_vec = NAS(HU_range,η_range,x,y,iteration1 , iteration2 , iteration3)
# plot(train_test_err_vec,title = "Train_error- test_error @ last iteration for each model")
##
""" Varying the MODEL itself """;


##
""" PYTHON CONVERTED CODE""";
Hidden_units = [2,3,2,5,1]
function weights(Hidden_units_perlayer,no_layers)
    no_layers
    global ld = Hidden_units_perlayer
    parameters1 = Dict()
    for l in 1 :4
        parameters1["W"*string(l)] = rand(ld[l+1],ld[l])
        parameters1["b"*string(l)] = zeros(ld[l+1])
        # display(ld[i])
        # display(rand(ld[i+1],ld[i]))
    end
    return parameters1
end
f12 = weights(Hidden_units, 4)
f12["W2"]
display(f12)
