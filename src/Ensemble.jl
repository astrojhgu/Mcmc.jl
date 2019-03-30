module Ensemble

using ..Utils
const draw_z = Utils.draw_z

using Random

function sample(logprob::Function,
    ensemble::AbstractArray{U,1},
    lp_cache::AbstractArray{T,1},
    a::T,
    rng = Random.GLOBAL_RNG) where
    {T <: AbstractFloat, U}
    nwalkers = length(ensemble)

    if nwalkers % 2 != 0
        error("number of walkers must be even")
    end
    ndims = length(first(ensemble))

    half_nwalkers = nwalkers ÷ 2

    walker_group_id = Random.shuffle(rng, [fill(0, half_nwalkers)...; fill(1, half_nwalkers)...]);

    walker_group = [findall(walker_group_id .== 0).-1, findall(walker_group_id .== 1).-1]
    rvec = Random.rand(rng, nwalkers)
    jvec = Random.rand(0:half_nwalkers-1, nwalkers)
    zvec = map(x->draw_z(a, rng), 0:nwalkers-1)


    lp_cached = length(lp_cache)==length(ensemble)
    if !lp_cached
        resize!(lp_cache, nwalkers)
        for i in 0:nwalkers-1
            lp_cache[firstindex(lp_cache)+i] = logprob(ensemble[firstindex(ensemble)+i])
        end
    end

    new_ensemble = copy(ensemble)
    new_lp_cache = copy(lp_cache)

    for k in 0:nwalkers-1
        lp_last_y = lp_cache[firstindex(lp_cache)+k]
        i = walker_group_id[1+k]
        j = jvec[1+k]
        ni = 1 - i
        z = zvec[1+k]
        r = rvec[1+k]
        new_y = ensemble[firstindex(ensemble)+k] * z + ensemble[firstindex(ensemble)+walker_group[1+ni][1+j]] * (one(T) - z)
        lp_y = logprob(new_y)
        q = exp((ndims - one(T)) * log(z) + lp_y - lp_last_y);
        if r <= q
            new_ensemble[firstindex(new_ensemble)+k] = new_y
            new_lp_cache[firstindex(new_lp_cache)+k] = lp_y
        end
    end

    for i in eachindex(ensemble)
        ensemble[i]=new_ensemble[i]
    end

    for i in eachindex(lp_cache)
        lp_cache[i]=new_lp_cache[i]
    end
end

end
