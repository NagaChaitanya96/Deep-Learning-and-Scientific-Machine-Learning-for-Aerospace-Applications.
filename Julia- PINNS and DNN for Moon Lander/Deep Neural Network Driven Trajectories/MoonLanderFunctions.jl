""" Euler method""";
function EulerIntPert2(tfinal,n)
      time1 = collect(0:1/n:tfinal)
      t1= time1
      global t1
      global uncertanity = [(rand() >0.5) ? 1 : 0 for _ in 1:length(t1)];

      h = [AnalyticalSolRand2(a,v0,h0)[2] for a in time1]
      v = [AnalyticalSolRand2(a,v0,h0)[1] for a in time1]

      # u = model([h';v'])

      dt = time1[2]-time1[1]
      hnew = [h0]; vnew = [v0]; up =[];
      for i in 1:length(time1)-1
            u = reduce(hcat,model([last(hnew),last(vnew)]))
            hn = hnew[i] + dt * vnew[i]
            vn = vnew[i] + dt * (-1.5+u)
            hnew = push!(hnew,hn)
            vnew = push!(vnew, vn)
            up = push!(up,u)
      end
      return hnew,vnew,h,v
end


function EulerInt2(tfinal,n)
      time1 = collect(0:1/n:tfinal)
      t1= time1
      global t1
      h = [AnalyticalSol(a,v0,h0)[4] for a in time1]
      v = [AnalyticalSol(a,v0,h0)[3] for a in time1]

      # u = model([h';v'])

      dt = time1[2]-time1[1]
      hnew = [h0]; vnew = [v0]; up =[];
      for i in 1:length(time1)-1
            u = reduce(hcat,model([last(hnew),last(vnew)]))
            hn = hnew[i] + dt * vnew[i]
            vn = vnew[i] + dt * (-1.5+u)
            hnew = push!(hnew,hn)
            vnew = push!(vnew, vn)
            up = push!(up,u)
      end
      return hnew,vnew,h,v
end
""" Rk4 integration with perturbation inside the trajectory """;
function RK4integralwithPerturbed(tf,v0,h0,n)
    global t1 = LinRange(0,tf,Int64(n))
    dt = (t1[2]-t1[1]) #we are chosing dt = 2*dt because we have u values only for selected dt range
    dh = dt;
    global uncertanity = [(rand() >0.5) ? 1 : 0 for _ in 1:length(t1)];
    # for the added uncertanity we have the analytical solution at each timestep
    v_data = [AnalyticalSolRand2(a,v0,h0)[1] for a in t1]
    h_data = [AnalyticalSolRand2(a,v0,h0)[2] for a in t1]
    u_data = [AnalyticalSolRand2(a,v0,h0)[3] for a in t1]
    tmid = [(t1[i] + t1[i+1])/2 for i in 1:length(t1)-1]

    # calculating the values at mid points for RK integration method

    h_md = [AnalyticalSolRand2(a,v0,h0)[2] for a in tmid]
    v_md = [AnalyticalSolRand2(a,v0,h0)[1] for a in tmid]
    u_md = model([h_md v_md]')
    u_pred = model([h_data v_data]')

    # This gives us data for only 100 points that we selected but for the values RK4 integration we
    # need data points at u(i+1/2)
    # h[i+1] = h[i] + (1/6)*(dh*v[i] + 4*dh*v[i+1/2] + dh*v[i+1])

    hnew = [h_data[1]]
    vnew = [v_data[1]]
    vmidnew = [v_md[1]]
    for i in 1:length(t1)-1
        vn = vnew[i] + (1/6) * ( dh*( -1.5 + u_pred[i] ) + 4* dh *( -1.5 + u_md[i]) + dh*(-1.5 + u_pred[i+1]) )
        vnew = push!(vnew , vn)
        if i < length(t1)-1
            vmd = vmidnew[i] + (1/6) * ( dh*( -1.5 + u_md[i] ) + 4* dh *( -1.5 + u_pred[i+1]) + dh*(-1.5 + u_md[i+1]) )
            vmidnew = push!(vmidnew,vmd)
        end
        hn = hnew[i] + (1/6) * ( dh * vnew[i] + 4 * dh * vmidnew[i] + dh * vnew[i+1] )
        hnew = push!(hnew , hn)
    end
    return hnew,vnew,vec(u_pred),h_data,v_data,u_data
end
##
""" Analytical solution using uncertainty in trajectory""";
function AnalyticalSolRand2(t,v0,h0;uncertanity = uncertanity)
    for i in 1:length(uncertanity)
        tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
        ŝ = tf̂/2 + v0/3
        if uncertanity[i] == 0
            v = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
            h = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
            u = (t<=ŝ) ? 0 : 3
            #print("rand is  < 0.5")
            return v,h,u
        elseif uncertanity[i] == 1
            v = (t<=ŝ) ? (-1.5*t + v0)*1.01 : (1.5*t -3*ŝ+v0)*1.01
            h = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0)*1.01 : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)*1.01
            u = (t<=ŝ) ? 0 : 3
            #print("rand is  > 0.5")
            return v,h,u
        end
    end

end
##
""" Callback with storing train and test error """;
function cb!()
    push!(err_train,loss(x11,y11))
    push!(err_test, loss(xt1,yt1))
    println("loss of the function is :" ,loss(x11,y11))
end
##
""" MSE loss function """
function loss_mse(a,b)
    Flux.Losses.mse(model(a),b)
end
function loss_bce(a,b)
    Flux.Losses.binarycrossentropy
end
##
""" Final time finding and analytical solution for ML""";
tfinal_(h0,v0) = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
function AnalyticalSol(t,v0,h0)
    tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
    ŝ = tf̂/2 + v0/3
    v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
    ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
    û(t) = (t<=ŝ) ? 0 : 3
    return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
end
##
""" Rk4 integrartion without pert"""
function RK4integral(tf,v0,h0,n)
    global t1 = LinRange(0,tf,Int64(n))
    dt = (t1[2]-t1[1]) #we are chosing dt = 2*dt because we have u values only for selected dt range
    dh = dt;
    h_data = [AnalyticalSol(a,v0,h0)[4] for a in t1]
    v_data = [AnalyticalSol(a,v0,h0)[3] for a in t1]
    u_data = [AnalyticalSol(a,v0,h0)[5] for a in t1]

    tmid = [(t1[i] + t1[i+1])/2 for i in 1:length(t1)-1]

    # calculating the values at mid points for RK integration method

    h_md = [AnalyticalSol(a,v0,h0)[4] for a in tmid]
    v_md = [AnalyticalSol(a,v0,h0)[3] for a in tmid]
    u_md = model([h_md v_md]')
    u_pred = model([h_data v_data]')

    # This gives us data for only 100 points that we selected but for the values RK4 integration we
    # need data points at u(i+1/2)
    # h[i+1] = h[i] + (1/6)*(dh*v[i] + 4*dh*v[i+1/2] + dh*v[i+1])

    hnew = [h_data[1]]
    vnew = [v_data[1]]
    vmidnew = [v_md[1]]
    for i in 1:length(t1)-1
        vn = vnew[i] + (1/6) * ( dh*( -1.5 + u_pred[i] ) + 4* dh *( -1.5 + u_md[i]) + dh*(-1.5 + u_pred[i+1]) )
        vnew = push!(vnew , vn)
        if i < length(t1)-1
            vmd = vmidnew[i] + (1/6) * ( dh*( -1.5 + u_md[i] ) + 4* dh *( -1.5 + u_pred[i+1]) + dh*(-1.5 + u_md[i+1]) )
            vmidnew = push!(vmidnew,vmd)
        end
        hn = hnew[i] + (1/6) * ( dh * vnew[i] + 4 * dh * vmidnew[i] + dh * vnew[i+1] )
        hnew = push!(hnew , hn)
    end
    return hnew,vnew,vec(u_pred),h_data,v_data,u_data
end
##
""" Euler Integration """;
function EulerInt(tfinal,n)
      time1 = 0:1/n:tfinal
      h = [AnalyticalSol(a,v0,h0)[4] for a in time1]
      v = [AnalyticalSol(a,v0,h0)[3] for a in time1]
      u = model([h';v'])
      dt = time1[2]-time1[1]
      hnew = [h0]; vnew = [v0];
      for i in 1:length(time1)-1
            hn = hnew[i] + dt * vnew[i]
            vn = vnew[i] + dt * (-1.5+u[i])
            hnew = push!(hnew,hn)
            vnew = push!(vnew, vn)
      end
      return hnew,vnew,h,v
end
export RK4integralwithPerturbed,loss_bce , EulerInt, RK4integral, AnalyticalSolRand2, loss_mse, cb!
