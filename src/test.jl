include("mcmc.jl")

using Random
import SpecialFunctions

function phi(x::Float64)::Float64
    (1.0+SpecialFunctions.erf(x/sqrt(2.0)))/2.0
end

function logfrac(x::Float64)::Float64
    SpecialFunctions.lgamma(x+1.0)
end

function logcn(m::Float64, n::Float64)::Float64
    logfrac(m)-logfrac(n)-logfrac(m-n)
end

function logbin(x::Float64, p::Float64, n::Float64)::Float64
    logcn(n,x)+x*log(p)+(n-x)*log(1.0-p)
end

E=[2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 70.0, 38.0, 42.0, 46.0, 50.0, 54.0, 58.0,
        62.0, 66.0, 74.0, 78.0, 82.0, 86.0, 90.0, 94.0, 98.0, 34.0,]

nrec=[
        23.0, 71.0, 115.0, 159.0, 200.0, 221.0, 291.0, 244.0, 44.0, 221.0, 210.0, 182.0, 136.0,
        119.0, 79.0, 81.0, 61.0, 41.0, 32.0, 32.0, 31.0, 22.0, 18.0, 11.0, 277.0,
    ]

ninj=[
        96.0, 239.0, 295.0, 327.0, 345.0, 316.0, 349.0, 281.0, 45.0, 235.0, 217.0, 185.0, 140.0,
        121.0, 79.0, 81.0, 61.0, 41.0, 32.0, 32.0, 31.0, 22.0, 18.0, 11.0, 298.0,
    ];

function logprob(x::Array{Float64, 1})::Float64
    energy=E
    a,b,mu,sigma=x
    if a<0.0 || a>1.0 || b<0.0 || b>1.0 || mu <0.0 || mu >100.0 || sigma<1e-6 || sigma>100.0
        #println((a,b,mu,sigma))
        -Inf64
    else
        logprob=0.0
        for (e, r, i) in zip(energy, nrec, ninj)
            eff=a+(b-a)*phi((e-mu)/sigma)
            logprob+=logbin(r, eff, i)
        end
        logprob
    end
end

let 
ensemble=empty([], Array{Float64,1})
for i in 1:16
    a=Random.rand()*0.1
    b=Random.rand()*0.1+0.89
    mu=Random.rand()*1.0+15
    sigma=Random.rand()*3.0+10
    push!(ensemble, [a,b,mu,sigma])
end

lp=missing

beta_list=map(x->2.0^(-x), 0:3)


import Plots

Plots.pyplot()

hist=empty([], Array{Float64, 1})

for i in 1:1000
    ensemble, lp=mcmc.ptsample.sample(logprob, ensemble, lp, beta_list, true, 0.5)
    #push!(hist, ensemble[1])
end


for i in 1:30000
    ensemble, lp=mcmc.ptsample.sample(logprob, ensemble, lp, beta_list, true, 0.5)
    push!(hist, ensemble[1])
end

d=transpose(hcat(hist...))
Plots.scatter(d[:,1], d[:,2], markersize=0.5,label="a")
Plots.savefig("a.png")
end
