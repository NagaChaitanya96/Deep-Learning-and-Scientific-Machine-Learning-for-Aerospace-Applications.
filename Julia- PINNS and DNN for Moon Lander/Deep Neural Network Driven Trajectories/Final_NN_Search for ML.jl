""" NEURAL ARCHITECCTURE SEARCH """;
cd(@__DIR__)
using Random
using Flux
using BSON: @save, @load
using Plots
""" Analytical solution details""";
v0= -2.; h0 = 10.
include("MoonLanderFunctions.jl")
include("AutoML2.jl")
##
""" Data generation for ML problem """;
""" Functions """;
n1 = [1 5 10];
dists = [(-a:1:a)*0.01 for a in n1];
num = [100 500 1000];
x11,y11,xt1,yt1 =[],[],[],[];
idx=[];
""" Generating test data for error approximation """;
htest,vtest = 10.015,-2.065 #vec([maximum(h_dist)+0.0015*a for a in [1 5 10]]) , vec([minimum(v_dist)-0.0015*a for a in [1 5 10]]);
step2 = 100
tftest = tfinal_(htest,vtest)
htd = [AnalyticalSol(b,vtest,htest)[4] for b in LinRange(0,tftest,step2)]
vtd = [AnalyticalSol(b,vtest,htest)[3] for b in LinRange(0,tftest,step2)]
utd = [AnalyticalSol(b,vtest,htest)[5] for b in LinRange(0,tftest,step2)]
all_data = [];
test_all =[];
utd1 =[];
err_train= [];
err_test= [];
train_test_err_vec = [];
train_loss_vec = [];
test_loss_vec = [];
model_vec = [];
best_model_vec = [];
lowest_err_vec = [];
hyper_param_sets_used = [];
##
""" Utilizing the Autogenerateed ML functions to find the best possible architecture for our problem """;
HU_range = [2^(n) for n in 1:5]
η_range = [10^r for r in LinRange(-3,-0.31,9)];
iteration1 = 10
##
""" Dont change M because we are not checking for robustness now""";
M = 1;  # number of times to check ROBUSTNESS ASSUMED OUR MODEL IS ROBUST ENOUGH.
##
N = 7500;
for i in 1: iteration1
        η = round(η_range[randperm(length(η_range))[1]]*rand(),digits = 6)
        n = HU_range[randperm(length(HU_range))[1]]
        display([η n])
        push!(hyper_param_sets_used,[η n])
        # now for this particular hyper params out model will run length(x1) times and generate plots
        for step1 in num
                for a in n1
                        #0.01*(-2:0.5:2)
                        dist = (-1:0.2:1)*0.01*a
                        global v_dist,h_dist = v0.+collect(dist), h0.+collect(dist)
                        tf_dist = [(2/3)*(v1)+ (4/3*(sqrt((v1^2/2) + 1.5*h1))) for v1 in v_dist, h1 in h_dist]
                        h_matrix = [];
                        u_matrix = [];
                        v_matrix = [];
                        global h_mat,v_mat,u_mat,t_all
                        h_mat = [];
                        v_mat = [];
                        u_mat = [];
                        t_all = [];

                        for v0 in v_dist
                                for h0 in h_dist
                                        tfinal = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
                                        h_data = [AnalyticalSol(a,v0,h0)[4] for a in LinRange(0,tfinal,step1)]
                                        v_data = [AnalyticalSol(a,v0,h0)[3] for a in LinRange(0,tfinal,step1)]
                                        u_data = [AnalyticalSol(a,v0,h0)[5] for a in LinRange(0,tfinal,step1)]

                                        h_matrix = push!(h_matrix,h_data)
                                        v_matrix = push!(v_matrix,v_data)
                                        u_matrix = push!(u_matrix,u_data)

                                        h_mat = hcat(h_matrix...)
                                        v_mat = hcat(v_matrix...)
                                        u_mat = hcat(u_matrix...)
                                        t_all = push!(t_all,tfinal)
                                end
                        end

                        #t_all == vec(tf_dist') # this states that for each combination of v0,h0 we have enough data and that is true.
                        h_data  = vec(h_mat)
                        v_data  = vec(v_mat)
                        u_data  = vec(u_mat)
                        global X_T = [h_data' ; v_data']
                        global Y_T = hcat(u_data...)

                        """ Until Here we will get data for each value of variation from 1% to 10 % and number
                        of points varying in between. Now for this variation we find the best possible trajectory
                        by inluding our code here and find the best_model_vec and analyse it. """;
                        global X = X_T'
                        global Y = Y_T'
                        # [h_mat v_mat u_mat]
                        """ Now we are goind to test 10 models having randomly initialized η and n values and take the
                        best model which gives us robustness and fixed number of iterations i.e 3000 """;

                        # here best_model_vec gives us many same models. So we give unique to get different models
                         # no of iterations in Optimizaiton.
                        """ We need to train 1 particular model for all the above variations and then change our model
                        For that we need to take out the η ,n and iteration1 term out of NAS  """;
                        display([η n step1 a])

                        train_test_err_vec,train_loss_vec,test_loss_vec,model_vec,lowest_err_vec,
                                best_model_vec = NAS2(n,η,X,Y,M,N,step1,a)

                        @save "model_vec2.bson" model_vec2

                end
        end

end
##
# Total number of matrix size should be length(x11)* Iter1*iter2 = 10 * 1*9 = 90 models
er_train1 = reshape(err_train,N,length(x11))
er_test1 = reshape(err_test,N,length(x11))

""" This error plot defines how well our neural network is training for that particular training data set""";
for i in 1:iteration1
        sections = [9*(i-1)+1:9*(i)]
        display(sections)
        p1 = plot()
        for j in sections[1]
                plot!(log10.(er_train1)[:,j],xlabel = :"Iterations",ylabel = "log10(error)",#yaxis = :log10,
                label = "$(idx[j][1]) steps, $(idx[j][2])% variation ",title = "TRAIN SET ERROS η = $(hyper_param_sets_used[i][1]), with $(hyper_param_sets_used[i][2]) Hidden
                units Iterations = $(N)",legend= :best)
                display(p1)
        end
        f1 = string(pwd())*"\\Train; Learning Rate  $(hyper_param_sets_used[i][1]) Iterations = $(N) $(hyper_param_sets_used[i][2]) Hidden units"
        png(f1)
end
""" FOR TEST DATA """;
for i in 1:iteration1
        sections = [9*(i-1)+1:9*(i)]
        display(sections)
        p2 = plot()
        for j in sections[1]
                plot!(log10.(er_test1)[:,j],xlabel = :"Iterations",ylabel = "log10(error)",#yaxis = :log10,
                label = "$(idx[j][1]) steps, $(idx[j][2])% variation ",title = "TEST SET ERRORS ,η = $(hyper_param_sets_used[i][1]), with $(hyper_param_sets_used[i][2]) Hidden
                units Iterations = $(N)",legend= :best)
                display(p2)
        end
        f2 = string(pwd())*"\\Test; Learning Rate  $(hyper_param_sets_used[i][1]) Iterations = $(N) $(hyper_param_sets_used[i][2]) Hidden units"
        png(f2)
end

for k in 1:iteration1
        for j in 1:3
                p3 = plot()
                for i in j:3:9
                        p3 = plot!(log10.(er_train1)[:,i],xlabel = :"Iterations",ylabel = "log10(error)",#yaxis = :log10,
                        label = "$(idx[i][1]) steps, $(idx[i][2])% variation ",
                        title = "With $(idx[j][2])% variation η = $(hyper_param_sets_used[k][1]),
                with $(hyper_param_sets_used[k][2]) Hidden units
                and Iterations = $(N)",legend= :best)
                end
                display(p3)
                f3 = string(pwd())*"\\ $(idx[j][2])% variation Learning Rate $(hyper_param_sets_used[k][1]) Iterations = $(N) $(hyper_param_sets_used[k][2]) Hidden units"
                png(f3)
        end
end
best_mv = unique(best_model_vec)
@save "best_model_vec.bson" best_mv
##
plot(sort(vec(best_mv[1](X'))))
plot!(sort(vec(best_mv[2](X'))))
plot!(sort(vec(best_mv[3](X'))))
plot!(sort(vec(Y)))

model = best_model_vec[66]
# plot!(best_mv[2](X'))
# plot!(best_mv[3](X'))
# plot!(Y)

loss2(a,b) = Flux.Losses.mse(model(a),b)
loss2(X',Y')


sort(vec(best_mv[1](X')))

scatter(sort(vec(best_mv[2](X'))))
