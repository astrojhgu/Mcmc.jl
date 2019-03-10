module Utils
using Random

function draw_z(a::T, rng::R) where
    {T <: AbstractFloat,R <: Random.AbstractRNG}
    sqrt_a = sqrt(a)
    unit = one(T)
    two = unit + unit
    p = Random.rand(rng, T) * two * (sqrt_a - unit / sqrt_a)
    y = unit / sqrt_a + p / two
    y * y
end

end
