module Ensemble

using ..Utils
const draw_z = Utils.draw_z

using Random

function sample(logprob::Function, 
    ensemble::Array{Array{T,1},1}, 
    lp_cache::Array{T,1}, 
    a::T, 
    rng = Random.GLOBAL_RNG) where 
    {T <: AbstractFloat}
    nwalkers = length(ensemble)

    if nwalkers % 2 != 0
        error("number of walkers must be even")
    end
    ndims = length(first(ensemble))

    half_nwalkers = nwalkers ÷ 2

    walker_group_id = Random.shuffle(rng, [fill(1, half_nwalkers)...; fill(2, half_nwalkers)...]);

    walker_group = [findall(walker_group_id .== 1), findall(walker_group_id .== 2)]
    rvec = Random.rand(rng, nwalkers)
    jvec = Random.rand(1:half_nwalkers, nwalkers)
    zvec = map(x->draw_z(a, rng), 1:nwalkers)


    lp_cached = length(lp_cache)==length(ensemble)
    if !lp_cached
        resize!(lp_cache, nwalkers)
        for i in 1:nwalkers
            lp_cache[i] = logprob(ensemble[i])
        end
    end

    new_ensemble = copy(ensemble)
    new_lp_cache = copy(lp_cache)

    for k in 1:nwalkers
        lp_last_y = lp_cache[k]
        i = walker_group_id[k]
        j = jvec[k]
        ni = 3 - i
        z = zvec[k]
        r = rvec[k]
        new_y = ensemble[k] * z + ensemble[walker_group[ni][j]] * (one(T) - z)
        lp_y = logprob(new_y)
        q = exp((ndims - one(T)) * log(z) + lp_y - lp_last_y);
        if r <= q 
            new_ensemble[k] = new_y
            new_lp_cache[k] = lp_y
        end
    end
    for i in 1:nwalkers
        ensemble[i]=new_ensemble[i]
        lp_cache[i]=new_lp_cache[i]
    end
end

end
