x = -1:0.01:1
Tn(n,x) = cos.(n*acos.(x))
plot(title = "Chebyshev polynomials",xlabel = "x range from -1 to 1", ylabel = " Tₙ(x) ")
for n in 0:7
    display(plot!(x,Tn(n,x), label = "Tₙ($(n))"))
end
png("D:\\2022\\January 2022")

# λᵣ * ṙ + λᵥ * v̇+ λₘ * ṁ + dJ/dt
# ṙ, v̇, ṁ
#=

θ = -π:0.01*π: π
x1 = cos.(θ)
function Tn(n)
return[ cos(n*x) for x in θ]
end

plot(xlabel = "x range from -1 to 1", ylabel = "Chebyshev polynomials Tₙ(x) ")
for n in 0:7
display(plot!(x1,Tn(n), label = "Tₙ($(n))"))
end
plot(x1,Tn(7), label = "Tₙ($(7))")
