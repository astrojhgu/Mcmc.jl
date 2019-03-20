module Pt

using ..Utils
using Random

const draw_z=Utils.draw_z

function pt_only_sample(logprob::Function, 
    ensemble::AbstractArray{U,1},
    lp_cache::AbstractArray{T,1},
    beta_list::AbstractArray{T,1},
    a::T, 
    rng = Random.GLOBAL_RNG) where 
    {T <: AbstractFloat, U}
    
    nbetas = length(beta_list)
    

    nwalkers = length(ensemble) ÷ nbetas

    if nwalkers % 2 != 0
        error("number of walkers must be even")
    end

    if nbetas * nwalkers != length(ensemble)
        error("nbeta*nwalkers!=len(ensenble)")
    end
    
    ndims = length(first(ensemble))

    half_nwalkers = nwalkers ÷ 2

    walker_group_id = map(x->Random.shuffle(rng, [fill(0, half_nwalkers)...; fill(1, half_nwalkers)...]), 1:nbetas)

    walker_group = map(x->[findall(x .== 0).-1, findall(x .== 1).-1], walker_group_id)
    rvec = map(x->Random.rand(rng, T, nwalkers), 1:nbetas)
    jvec = map(x->Random.rand(0:half_nwalkers-1, nwalkers), 1:nbetas)
    zvec = map(x->map(x->draw_z(a, rng), 1:nwalkers), 1:nbetas)


    lp_cached = length(lp_cache)==length(ensemble)
    if !lp_cached
        resize!(lp_cache, nwalkers*nbetas)
        #lp_cache = fill(zero(T), nwalkers * nbetas)
        for i in 0:nwalkers * nbetas-1
            lp_cache[firstindex(lp_cache)+i] = logprob(ensemble[firstindex(ensemble)+i])
        end
    end

    new_ensemble = copy(ensemble)
    new_lp_cache = copy(lp_cache)

    for n in 0:nwalkers * nbetas-1
        ibeta = n ÷ nwalkers
        k = n - ibeta * nwalkers
        lp_last_y = lp_cache[firstindex(lp_cache)+ibeta * nwalkers + k]

        i = walker_group_id[1+ibeta][1+k]
        j = jvec[1+ibeta][1+k]
        ni = 1 - i
        z = zvec[1+ibeta][1+k]
        r = rvec[1+ibeta][1+k]
        new_y = ensemble[firstindex(ensemble)+ibeta * nwalkers + k] * z + ensemble[firstindex(ensemble)+ibeta * nwalkers + walker_group[1+ibeta][1+ni][1+j]] * (one(T) - z)
        lp_y = logprob(new_y)
        beta = beta_list[firstindex(beta_list)+ibeta]
        delta_lp = lp_y - lp_last_y
        q = exp((ndims - one(T)) * log(z) + delta_lp * beta);
        if r <= q 
            new_ensemble[firstindex(ensemble)+ibeta * nwalkers + k] = new_y
            new_lp_cache[firstindex(new_lp_cache)+ibeta * nwalkers + k] = lp_y
        end
    end

    for i in eachindex(ensemble)
        ensemble[i]=new_ensemble[i]
        lp_cache[i]=new_lp_cache[i]
    end
    #(ensemble, lp_cache)
end

function exchange_prob(lp1::T, lp2::T, beta1::T, beta2::T)::T where {T <: AbstractFloat}
    x = exp((beta2 - beta1) * (-lp2 + lp1))
    if x > one(T)
        one(T)
    else
        x
    end
end

function swap_element(x::A, i::I, j::I) where {A <: AbstractArray,I <: Integer}
    x[i], x[j] = x[j], x[i]
end


function swap_walkers(ensemble, lp_cache, beta_list::Array{T,1}, rng = Random.GLOBAL_RNG) where {T <: AbstractFloat}
    nbeta = length(beta_list)
    nwalkers_per_beta = length(ensemble) ÷ nbeta
    if nwalkers_per_beta * nbeta != length(ensemble)
        error("nwalkers % nbeta!=0")
    end

    if length(lp_cache)!=length(ensemble)
        return
    end

    for i in nbeta-1:-1:1
        beta1 = beta_list[firstindex(beta_list)+i]
        beta2 = beta_list[firstindex(beta_list)+i - 1]
        if beta1 >= beta2 
            error("Error, beta list must be in desc order")
        end
        jvec = Random.shuffle(rng, [0:nwalkers_per_beta-1; ])
        for j in 0:nwalkers_per_beta-1
            j1 = jvec[1+j]
            j2 = j
            lp1 = lp_cache[firstindex(lp_cache)+i * nwalkers_per_beta + j1]
            lp2 = lp_cache[firstindex(lp_cache)+(i - 1) * nwalkers_per_beta + j2]
            r = Random.rand(rng, T)
            ep = exchange_prob(lp1, lp2, beta1, beta2)
            if r < ep
                swap_element(ensemble, firstindex(ensemble)+i * nwalkers_per_beta + j1, firstindex(ensemble)+(i - 1) * nwalkers_per_beta + j2)
                swap_element(lp_cache, firstindex(ensemble)+i * nwalkers_per_beta + j1, firstindex(ensemble)+(i - 1) * nwalkers_per_beta + j2)
            end
        end        
    end
end

function sample(logprob::Function, 
    ensemble::Array{Array{T,1},1}, 
    lp_cache::Array{T,1}, 
    beta_list::Array{T,1}, 
    perform_swap::Bool, 
    a::T, 
    rng = Random.GLOBAL_RNG) where {T <: AbstractFloat}
    if perform_swap
        swap_walkers(ensemble, lp_cache, beta_list, rng)
    end
    pt_only_sample(logprob, ensemble, lp_cache, beta_list, a, rng)
end


end
