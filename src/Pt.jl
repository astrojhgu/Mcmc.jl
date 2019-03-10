module Pt

using ..Utils
using Random

const draw_z=Utils.draw_z

function pt_only_sample(logprob::Function, 
    ensemble::Array{Array{T,1},1}, 
    lp_cache::Array{T,1}, 
    beta_list::Array{T,1},
    a::T, 
    rng = Random.GLOBAL_RNG) where 
    {T <: AbstractFloat}
    
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

    walker_group_id = map(x->Random.shuffle(rng, [fill(1, half_nwalkers)...; fill(2, half_nwalkers)...]), 1:nbetas)

    walker_group = map(x->[findall(x .== 1), findall(x .== 2)], walker_group_id)
    rvec = map(x->Random.rand(rng, T, nwalkers), 1:nbetas)
    jvec = map(x->Random.rand(1:half_nwalkers, nwalkers), 1:nbetas)
    zvec = map(x->map(x->draw_z(a, rng), 1:nwalkers), 1:nbetas)


    lp_cached = length(lp_cache)==length(ensemble)
    if !lp_cached
        resize!(lp_cache, nwalkers*nbetas)
        #lp_cache = fill(zero(T), nwalkers * nbetas)
        for i in 1:nwalkers * nbetas
            lp_cache[i] = logprob(ensemble[i])
        end
    end

    new_ensemble = copy(ensemble)
    new_lp_cache = copy(lp_cache)

    for n in 1:nwalkers * nbetas
        ibeta = (n - 1) ÷ nwalkers + 1
        k = n - (ibeta - 1) * nwalkers
        lp_last_y = lp_cache[(ibeta - 1) * nwalkers + k]

        i = walker_group_id[ibeta][k]
        j = jvec[ibeta][k]
        ni = 3 - i
        z = zvec[ibeta][k]
        r = rvec[ibeta][k]
        new_y = ensemble[(ibeta - 1) * nwalkers + k] * z + ensemble[(ibeta - 1) * nwalkers + walker_group[ibeta][ni][j]] * (one(T) - z)
        lp_y = logprob(new_y)
        beta = beta_list[ibeta]
        delta_lp = lp_y - lp_last_y
        q = exp((ndims - one(T)) * log(z) + delta_lp * beta);
        if r <= q 
            new_ensemble[(ibeta - 1) * nwalkers + k] = new_y
            new_lp_cache[(ibeta - 1) * nwalkers + k] = lp_y
        end
    end

    for i in 1:nwalkers*nbetas
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

    for i in nbeta:-1:2
        beta1 = beta_list[i]
        beta2 = beta_list[i - 1]
        if beta1 >= beta2 
            error("Error, beta list must be in desc order")
        end
        jvec = Random.shuffle(rng, [1:nwalkers_per_beta; ])
        for j in 1:nwalkers_per_beta
            j1 = jvec[j]
            j2 = j
            lp1 = lp_cache[(i - 1) * nwalkers_per_beta + j1]
            lp2 = lp_cache[(i - 2) * nwalkers_per_beta + j2]
            r = Random.rand(rng, T)
            ep = exchange_prob(lp1, lp2, beta1, beta2)
            if r < ep
                swap_element(ensemble, (i - 1) * nwalkers_per_beta + j1, (i - 2) * nwalkers_per_beta + j2)
                swap_element(lp_cache, (i - 1) * nwalkers_per_beta + j1, (i - 2) * nwalkers_per_beta + j2)
            end
        end        
    end
end

function sample(logprob::Function, 
    ensemble::Array{Array{T,1},1}, 
    lp_cache::Union{Missing,Array{T,1}}, 
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