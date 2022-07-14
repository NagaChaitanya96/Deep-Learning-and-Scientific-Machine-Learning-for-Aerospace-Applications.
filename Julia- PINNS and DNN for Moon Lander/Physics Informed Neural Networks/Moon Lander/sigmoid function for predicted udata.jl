""" Code for writing sigmoid function and obtaining the parmeters for the corresponding data""";
begin
      v0= -2.; h0 = 10.
      tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
      tf = tf̂
end

function AnalyticalSol(t,v0,h0)
      tf̂ = (2/3)*(v0)+ (4/3*(sqrt((v0^2/2) + 1.5*h0)))
      ŝ = tf̂/2 + v0/3
      v̂(t) = (t<=ŝ) ? (-1.5*t + v0) : (1.5*t -3*ŝ+v0)
      ĥ(t) = (t<=ŝ) ? (-0.75*t^2 +v0*t +h0) : (.75*t^2+ (-3*ŝ +v0)*t + 1.5(ŝ)^2+h0)
      û(t) = (t<=ŝ) ? 0 : 3
      #println("The values are : $([tf̂, ŝ ,v̂(t) , ĥ(t), û(t)])")
      return tf̂, ŝ ,v̂(t) , ĥ(t), û(t)
end

hdata = [AnalyticalSol(a,-2.,10.)[4] for a in ts]
udata =  [AnalyticalSol(a,-2.,10.)[5] for a in ts]
vdata = [AnalyticalSol(a,-2.,10.)[3] for a in ts]
u_ = [hdata'; vdata'; udata']
t_ = hcat(collect(ts)...)
len = length(t_)

function sigmoidAPprox(a,b)
      σ1(t,a,b) = 3*1/(1+exp(-a*t+b))
      val = reduce(vcat,@. σ1(t_,a,b))
      plot(val)
      plot!(udata)
end
@manipulate for a in 0:1:200, b in 0:01:250
      sigmoidAPprox(a,b)
end


# a = 157; b = 221 for exact solution parameters
